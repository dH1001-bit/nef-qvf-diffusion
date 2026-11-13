#!/usr/bin/env python3
"""Evaluate a trained jernn network on its validation rows.

    python3 scripts/eval_nn.py --model fig/nn_pt_eta/best.pt
    python3 scripts/eval_nn.py --model ... --at receta:abs=0.25,1.25,2.75 --global fig/global_K3/global_fits.json
    python3 scripts/eval_nn.py --model ... --diffusion

The rows are read again with the training's own arguments (files, cuts, weights, fraction, seed),
so the validation split is the same; --files overrides the files (e.g. another sample, or real data:
then all rows are used).  Inputs are only read.

Outputs in --outdir (default <dir of --model>/eval):
  pit.png          PIT = F(y | x) of every row, weighted histogram, overall and in quantile slices of
                   every condition.  Flat (within the band) if the conditional density is right;
                   needs no truth beyond y itself, so it works on real data too.
  curves_<k>.png   median and sigma_68/median of y versus the first condition (--xcond), at fixed
                   values of the others (--at; default their validation medians).  Data: rows within
                   --window of the fixed values, in quantile bins of the x condition.  Network: the
                   same rows mixed (open squares; the like-for-like comparison) and the smooth curve
                   at the exact point (line).  With --global (needs genpt:log as the x condition and
                   an |eta| condition): the global fit's curve of the |eta| bin containing the point.
  diffusion.png    (--diffusion) the closed-form noised density p_t(y | x) of the network at the
                   first --at point for several t, against validation rows near that point noised
                   forward with the exact Ornstein-Uhlenbeck step: the ground truth a diffusion model
                   would be trained against.
  eval.txt         validation nll, PIT summary (KS distance), curve values.
"""
import argparse, itertools, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from jernn import data as jd                                          # noqa: E402
from jernn.diffusion import forward_sample, noised_pdf                # noqa: E402
from jernn.features import Cut, Feature                               # noqa: E402
from jernn.model import Predictor                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="checkpoint of train_nn.py")
ap.add_argument("--files", nargs="+", default=None, help="other files (default: the training's)")
ap.add_argument("--max-eval", type=int, default=300000, help="validation rows used (random subset)")
ap.add_argument("--xcond", type=int, default=0, help="index of the condition on the x axis of the curves")
ap.add_argument("--at", nargs="*", default=[], help="spec=v1,v2,...: fixed values (transformed) of other conditions")
ap.add_argument("--window", nargs="*", default=[], help="spec=half-width of the data window (default 0.2 sd)")
ap.add_argument("--xbins", type=int, default=20)
ap.add_argument("--slices", type=int, default=4, help="quantile slices per condition in the PIT plot")
ap.add_argument("--global", dest="glob", default=None, help="global_fits.json to overlay")
ap.add_argument("--diffusion", action="store_true")
ap.add_argument("--times", nargs="+", type=float, default=[0.0, 0.05, 0.2, 0.5, 1.5])
ap.add_argument("--device", default="cpu")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

P = Predictor.load(a.model, a.device)
info = P.info
A = info["args"]
outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(a.model)), "eval")
os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(a.seed)

# ------------------------------------------------------------------ rows (read only)
same = a.files is None
D = jd.read(a.files or info["data"]["files"], P.target, P.conds, [Cut(s) for s in info["cuts"]],
            weight=A["weight"], fraction=A["fraction"], max_rows=A["max_rows"], seed=A["seed"])
if same:
    _, rows = jd.split(len(D["y"]), A["val_frac"], A["seed"])
else:
    rows = np.arange(len(D["y"]))
if len(rows) > a.max_eval:
    rows = np.sort(rng.choice(rows, a.max_eval, replace=False))
y, X, w = D["y"][rows].astype(np.float64), D["X"][rows].astype(np.float64), D["w"][rows]
names = [f.spec for f in P.conds]
print("%d evaluation rows (%s), n_eff %.0f" % (len(y), "validation split" if same else "all rows", jd.n_eff(w)))

L = ["# eval_nn.py %s" % " ".join(sys.argv[1:]),
     "# model: target %s, conditions %s, trained %d epochs, val nll %.6f" % (info["target"], ", ".join(names),
                                                                           info["epoch"], info["val_nll"])]
lp = P.log_prob(y.astype(np.float32), X.astype(np.float32))
L.append("nll per unit weight on these rows: %.6f" % (-(w * lp).sum() / np.abs(w).sum()))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

# ------------------------------------------------------------------ PIT
u = P.pit(y, X)
ylo, yhi = (P.model.y_range or (-np.inf, np.inf))
ins = (y >= ylo) & (y < yhi)    # censored rows are left out (their tail mass is ~1e-3: negligible here)
nb = 20
edges = np.linspace(0, 1, nb + 1)


def pit_hist(ax, m, title):
    W = np.abs(w[m]); h = np.histogram(u[m], edges, weights=W)[0]
    h2 = np.histogram(u[m], edges, weights=W * W)[0]
    tot = W.sum()
    ax.bar(edges[:-1], h / tot * nb, width=1 / nb, align="edge", color="C0", alpha=0.6)
    ax.errorbar(0.5 * (edges[1:] + edges[:-1]), h / tot * nb, np.sqrt(h2) / tot * nb, fmt="none", color="k", lw=0.8)
    ax.axhline(1, color="C3", lw=1)
    ne = jd.n_eff(w[m])
    ax.fill_between([0, 1], 1 - np.sqrt(nb / ne), 1 + np.sqrt(nb / ne), color="C3", alpha=0.15, lw=0)
    srt = np.argsort(u[m]); cu = np.cumsum(W[srt]) / tot
    ks = float(np.max(np.abs(cu - u[m][srt])))
    ax.set_title("%s\nKS %.4f, $n_{eff}$ %.0f" % (title, ks, ne), fontsize=8)
    ax.set_ylim(0.5, 1.5); ax.tick_params(labelsize=7)
    return ks


nc = len(names)
fig, axes = plt.subplots(nc + 1, a.slices, figsize=(3.0 * a.slices, 2.4 * (nc + 1)), squeeze=False)
ks0 = pit_hist(axes[0, 0], ins, "all rows in the y range")
L.append("PIT KS distance, all rows: %.4f" % ks0)
for ax in axes[0, 1:]:
    ax.axis("off")
for i, nm in enumerate(names):
    qs = np.quantile(X[:, i], np.linspace(0, 1, a.slices + 1))
    for s in range(a.slices):
        m = ins & (X[:, i] >= qs[s]) & (X[:, i] <= qs[s + 1])
        if m.sum() > 50:
            ks = pit_hist(axes[i + 1, s], m, "%s in [%.3g, %.3g]" % (nm, qs[s], qs[s + 1]))
            L.append("PIT KS %-14s [%.3g, %.3g]: %.4f" % (nm, qs[s], qs[s + 1], ks))
fig.suptitle("PIT of the network (flat = calibrated; band: $\\pm1\\sigma$ statistical)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "pit.png"), dpi=110); plt.close(fig)

# ------------------------------------------------------------------ curves


def parse(specs, default):
    out = {}
    for s in specs:
        k, _, v = s.partition("=")
        if k not in names:
            raise SystemExit("%r is not a condition (conditions: %s)" % (k, ", ".join(names)))
        out[names.index(k)] = [float(t) for t in v.split(",")]
    return {i: out.get(i, default(i)) for i in range(nc) if i != a.xcond}


at = parse(a.at, lambda i: [float(np.median(X[:, i]))])
win = {i: v[0] for i, v in parse(a.window, lambda i: [0.2 * float(X[:, i].std())]).items()}
combos = [dict(zip(at, c)) for c in itertools.product(*at.values())] if at else [{}]


def wquant(v, ww, q):
    s = np.argsort(v); c = np.cumsum(ww[s]); c = c / c[-1]
    return np.interp(q, c, v[s])


def summ(v, ww):
    q = wquant(v, ww, [0.16, 0.45, 0.5, 0.55, 0.84])
    med, s68 = q[2], 0.5 * (q[4] - q[0])
    ne = jd.n_eff(ww)
    e_med = np.sqrt(0.25 / ne) * (q[3] - q[1]) / 0.1           # sqrt(p(1-p)/n) / density
    return med, s68 / med, e_med, e_med * np.sqrt(2) / med     # rough error of s68/med


def mixture_summ(Xr, wr, nmax=1500):
    if len(Xr) > nmax:
        k = rng.choice(len(Xr), nmax, replace=False, p=np.abs(wr) / np.abs(wr).sum()); Xr, wr = Xr[k], np.ones(nmax)
    mu, sig, c = P.params(Xr.astype(np.float32))
    Fz = P._cdf_z(c)
    yg = np.linspace(np.min(mu - 8 * sig), np.max(mu + 8 * sig), 3000)
    F = np.zeros_like(yg)
    for i in range(len(mu)):
        F += np.abs(wr[i]) * np.interp((yg - mu[i]) / sig[i], P.ZGRID, Fz[i], left=0, right=1)
    F /= np.abs(wr).sum()
    q = np.interp([0.16, 0.5, 0.84], F, yg)
    return q[1], 0.5 * (q[2] - q[0]) / q[1]


G = None
if a.glob:
    from response_models import BornModel
    with open(a.glob) as fh:
        G = json.load(fh)
    eta_i = next((i for i, n in enumerate(names) if "eta" in n and "abs" in n), None)
    if names[a.xcond] != "genpt:log" or eta_i is None:
        print("--global needs genpt:log as the x condition and an |eta| condition; overlay skipped")
        G = None

xc = X[:, a.xcond]
xg = np.linspace(*np.quantile(xc, [0.002, 0.998]), 80)
for k, cmb in enumerate(combos):
    sel = np.ones(len(y), bool)
    for i, v in cmb.items():
        sel &= np.abs(X[:, i] - v) < win[i]
    lab = ", ".join("%s = %g" % (names[i], v) for i, v in cmb.items()) or "all"
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    # smooth network curve at the exact point
    Xg = np.zeros((len(xg), nc)); Xg[:, a.xcond] = xg
    for i, v in cmb.items():
        Xg[:, i] = v
    Q = P.quantiles(Xg.astype(np.float32), [0.16, 0.5, 0.84])
    axs[0].plot(xg, Q[:, 1], "C3-", label="network at the point")
    axs[1].plot(xg, 0.5 * (Q[:, 2] - Q[:, 0]) / Q[:, 1], "C3-")
    L.append("\n# curves at %s (data window %s)" % (lab, ", ".join("%g" % win[i] for i in cmb)))
    L.append("# %10s %8s %10s %10s %10s %10s %10s" % (names[a.xcond], "n_eff", "med data", "med net", "s/m data",
                                                     "s/m net", "pull med"))
    if sel.sum() > 200:
        qb = np.unique(np.quantile(xc[sel], np.linspace(0, 1, a.xbins + 1)))
        for j in range(len(qb) - 1):
            m = sel & (xc >= qb[j]) & (xc < qb[j + 1]) & ins
            if m.sum() < 30:
                continue
            xm = np.average(xc[m], weights=np.abs(w[m]))
            dmed, dsm, emed, esm = summ(y[m], np.abs(w[m]))
            nmed, nsm = mixture_summ(X[m], w[m])
            axs[0].errorbar(xm, dmed, emed, fmt="ko", ms=3, label="data" if j == 0 else None)
            axs[1].errorbar(xm, dsm, esm, fmt="ko", ms=3)
            axs[0].plot(xm, nmed, "s", color="C3", mfc="none", ms=5, label="network, mixed over the same rows" if j == 0 else None)
            axs[1].plot(xm, nsm, "s", color="C3", mfc="none", ms=5)
            L.append("  %10.4f %8.0f %10.5f %10.5f %10.5f %10.5f %10.2f" % (xm, jd.n_eff(w[m]), dmed, nmed, dsm, nsm,
                                                                          (nmed - dmed) / emed))
    if G is not None:
        ev = cmb.get(eta_i)
        f = next((f for f in G["fits"] if ev is not None and f["abs_eta"][0] <= ev < f["abs_eta"][1]), None)
        if f is not None:
            mod, th = BornModel.from_dict(f["model"])
            R = np.linspace(0, 2, 2001); rc = 0.5 * (R[1:] + R[:-1])
            xs = np.linspace(max(mod.x0, xg[0]), min(mod.x1, xg[-1]), 60)
            pg = mod.pdf_grid(th, np.exp(xs), rc)
            cdf = np.cumsum(pg, 1); cdf /= cdf[:, -1:]
            qg = np.array([np.interp([0.16, 0.5, 0.84], c_, rc) for c_ in cdf])
            gl = "global fit (K=%d), |eta| %.1f-%.1f" % (mod.K, *f["abs_eta"])
            axs[0].plot(xs, qg[:, 1], "C0--", label=gl)
            axs[1].plot(xs, 0.5 * (qg[:, 2] - qg[:, 0]) / qg[:, 1], "C0--")
    axs[0].set_ylabel("median of %s" % P.target.label); axs[1].set_ylabel("$\\sigma_{68}$ / median")
    for ax in axs:
        ax.set_xlabel(P.conds[a.xcond].label); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.suptitle("network vs data at %s" % lab, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "curves_%d.png" % k), dpi=110); plt.close(fig)

# ------------------------------------------------------------------ diffusion ground truth
if a.diffusion:
    cmb = combos[0]
    x0 = np.zeros(nc); x0[a.xcond] = float(np.median(xc))
    for i, v in cmb.items():
        x0[i] = v
    sel = np.ones(len(y), bool)
    for i in range(nc):
        sel &= np.abs(X[:, i] - x0[i]) < (win.get(i, 0.1 * X[:, i].std()))
    mu0, s0, c0 = P.params(x0[None, :].astype(np.float32))
    mus, sgs, _ = P.params(X[sel].astype(np.float32))
    yg = np.linspace(mu0[0] - 7 * s0[0], mu0[0] + 7 * s0[0], 600)
    fig, axs = plt.subplots(1, len(a.times), figsize=(3.2 * len(a.times), 3.4), sharey=False)
    for ax, t in zip(np.atleast_1d(axs), a.times):
        ax.plot(yg, noised_pdf(yg, mu0[0], s0[0], c0[0], t), "C3-", lw=1.3, label="closed form $p_t$")
        if sel.sum() > 30:
            yt = forward_sample(y[sel], mus, sgs, t, rng)
            ax.hist(yt, 50, range=(yg[0], yg[-1]), weights=np.abs(w[sel]), density=True, histtype="step",
                    color="k", label="rows noised forward")
        ax.set_title("t = %g" % t, fontsize=9); ax.set_xlabel(P.target.label); ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.suptitle("exact Ornstein-Uhlenbeck noising of the network density at %s (%d rows in the window)"
                 % (", ".join("%s=%.3g" % (n, v) for n, v in zip(names, x0)), sel.sum()), fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "diffusion.png"), dpi=110); plt.close(fig)

print("\n".join(L))
with open(os.path.join(outdir, "eval.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print("outputs in", outdir)
