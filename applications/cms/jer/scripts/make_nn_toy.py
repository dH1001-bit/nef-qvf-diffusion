#!/usr/bin/env python3
"""Toy rows with a known truth for jernn: the global fit's p(r | pt, |eta| bin) as the truth.

    python3 scripts/make_nn_toy.py --global fig/global_K3/global_fits.json --n 500000 --out fig/nn_toy/toy.npz
    python3 scripts/train_nn.py --files fig/nn_toy/toy.npz --weight none --cut --outdir fig/nn_toy/net
    python3 scripts/eval_nn.py --model fig/nn_toy/net/best.pt --at receta:abs=0.25,1.25,2.75 \
        --global fig/global_K3/global_fits.json

ln pt_gen uniform over the fit range (flat in ln pt: no weights needed), |eta| uniform over the
global fit's bins, receta with a random sign, r drawn from the global density of that pt and
|eta| bin.  The truth is piecewise constant in |eta| (one global fit per bin), so a network that
recovers it closely inside the bins and smooths only at the bin edges passes the closure test.
No weight column (as real data): train with --weight none, or just let the reader fall back.
The global fit json is only read.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from response_models import BornModel                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--global", dest="glob", required=True)
ap.add_argument("--n", type=int, default=500000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.path.abspath(a.out) == os.path.abspath(a.glob):
    raise SystemExit("--out must not be the input")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

with open(a.glob) as fh:
    G = json.load(fh)
fits = sorted(G["fits"], key=lambda f: f["eta_bin"])
rng = np.random.default_rng(a.seed)
R = np.linspace(0.0, 2.0, 4001)
rc = 0.5 * (R[1:] + R[:-1])
lo_eta, hi_eta = fits[0]["abs_eta"][0], fits[-1]["abs_eta"][1]
aeta = rng.uniform(lo_eta, hi_eta, a.n)
ie = np.searchsorted([f["abs_eta"][1] for f in fits], aeta, side="right")
ie = np.minimum(ie, len(fits) - 1)
pt, r = np.empty(a.n), np.empty(a.n)
for k, f in enumerate(fits):
    mod, th = BornModel.from_dict(f["model"])
    m = np.flatnonzero(ie == k)
    x = rng.uniform(mod.x0, mod.x1, len(m))
    pt[m] = np.exp(x)
    # inverse CDF on a grid of 200 ln pt nodes (the density varies slowly in ln pt)
    xn = np.linspace(mod.x0, mod.x1, 200)
    cdf = np.cumsum(mod.pdf_grid(th, np.exp(xn), rc), 1)
    cdf /= cdf[:, -1:]
    j = np.clip(np.round((x - mod.x0) / (mod.x1 - mod.x0) * 199).astype(int), 0, 199)
    u = rng.random(len(m))
    for jj in np.unique(j):
        s = j == jj
        r[m[s]] = np.interp(u[s], cdf[jj], rc)
    print("|eta| bin %d: %d rows" % (k, len(m)), flush=True)
np.savez(a.out, genpt=pt.astype(np.float32), recpt=(r * pt).astype(np.float32),
         receta=(aeta * rng.choice([-1, 1], a.n)).astype(np.float32), dR=np.zeros(a.n, np.float32))
print("written", a.out)
