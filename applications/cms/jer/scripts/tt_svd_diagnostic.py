#!/usr/bin/env python3
"""TT-SVD diagnostic of a global fit: is the amplitude tensor c_n(ln pt, |eta| bin) low rank?

Reads global_fits.json of fit_response_global.py (read only) and builds, for every |eta| bin e,
the normalised amplitude c(x, e) = c~(x, e) / ||c~(x, e)|| on a grid in x = ln pt_gen.  Then it
takes the SVD of every unfolding of the 3-index tensor:

    shapes  (n | x e)   : how many amplitude shapes (bond chi_1 between the Hermite site and the rest)
    pt      (x | n e)   : how many pt functions
    eta     (n x | e)   : how many |eta| modes (bond chi_2 to the |eta| site)

Two grids:
  * own range   : each |eta| bin on its own fitted ln pt range (only for the shapes unfolding:
                  columns of different bins need not share x)
  * common range: all bins on the overlap of their ranges (all three unfoldings)

|h|^2, and with it p(r | pt), does not change under (a) a phase per point, c -> e^{i phi} c, and
(b) flipping roots of h(z) = sum c_n phi_n(z) to their complex conjugates.  Neither is fixed by
the fit in a way that is consistent across |eta| bins, and both can raise the rank artificially.
Three gauges are compared:
    fit    : the fit's own phase (b_00 = 1), normalised
    c0real : phase per point such that c_0 is real >= 0
    roots  : all roots of h moved to the upper half plane, then c_0 real >= 0 (a canonical form)

Caveat: c lives in the reference frame z = (r - mu_e(pt)) / sigma_e(pt) of its own bin, so
equal c in two bins means equal shape relative to each bin's own Gaussian, not equal densities.

Outputs in --outdir (default <dir of --global>/tt_svd): tt_svd.txt, tt_svd.png.
"""
import argparse, json, os, sys
from math import factorial
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from response_models import BornModel

ap = argparse.ArgumentParser()
ap.add_argument("--global", dest="glob", default="fig/global_receta/global_fits.json")
ap.add_argument("--npts", type=int, default=60, help="grid points in ln pt per |eta| bin")
ap.add_argument("--nshow", type=int, default=8, help="singular values printed per unfolding")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()
outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(a.glob)), "tt_svd")
os.makedirs(outdir, exist_ok=True)

with open(a.glob) as fh:
    G = json.load(fh)
fits = sorted(G["fits"], key=lambda f: f["eta_bin"])
models = [BornModel.from_dict(f["model"]) for f in fits]
J = models[0][0].J
labels = ["%.1f-%.1f" % tuple(f["abs_eta"]) for f in fits]
SQF = np.sqrt([float(factorial(n)) for n in range(J + 1)])


def gauge(c, kind):
    """c: (npts, J+1) normalised rows -> the chosen gauge."""
    if kind == "fit":
        return c
    if kind == "roots":
        out = np.empty_like(c)
        for i, ci in enumerate(c):
            p = np.polynomial.hermite_e.herme2poly(ci / SQF)          # monomial coefficients, low -> high
            nz = np.flatnonzero(np.abs(p) > 1e-14 * np.abs(p).max())
            p = p[: nz[-1] + 1]
            rt = np.polynomial.polynomial.polyroots(p)
            rt = np.where(rt.imag < 0, rt.conj(), rt)
            q = p[-1] * np.polynomial.polynomial.polyfromroots(rt)
            h = np.zeros(J + 1, complex)
            hh = np.polynomial.hermite_e.poly2herme(q) * SQF[: len(q)]
            h[: len(hh)] = hh
            out[i] = h / np.linalg.norm(h)
        c = out
    return c * np.exp(-1j * np.angle(c[:, :1]))                        # c_0 real >= 0


def coeffs(mod, th, x):
    c, _, _ = mod.coefficients(th, np.exp(x))
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def sv(M):
    s = np.linalg.svd(M, compute_uv=False)
    return s / s[0], np.sqrt(np.maximum(1 - np.cumsum(s ** 2) / np.sum(s ** 2), 0))   # relative, truncation error


x0 = max(m.x0 for m, _ in models)
x1 = min(m.x1 for m, _ in models)
xc = np.linspace(x0, x1, a.npts)
GAUGES = ("fit", "c0real", "roots")
res = {}
for g in GAUGES:
    own = np.concatenate([gauge(coeffs(m, th, np.linspace(m.x0, m.x1, a.npts)), g) for m, th in models])  # (7 npts, n)
    T = np.stack([gauge(coeffs(m, th, xc), g) for m, th in models], axis=2)                               # (x, n, e)
    res[g] = {
        "shapes, own ranges (n | x e)": sv(own.T),
        "shapes (n | x e)": sv(T.transpose(1, 0, 2).reshape(J + 1, -1)),
        "pt (x | n e)": sv(T.reshape(a.npts, -1)),
        "eta (n x | e)": sv(T.reshape(-1, T.shape[2])),
    }
    # the |eta| bins alone: rank of each bin's (n | x) matrix, i.e. chi_1 needed bin by bin
    res[g]["per bin (n | x)"] = [sv(gauge(coeffs(m, th, np.linspace(m.x0, m.x1, a.npts)), g).T) for m, th in models]

L = ["# TT-SVD diagnostic of %s" % os.path.abspath(a.glob),
     "# J = %d, K = %s, %d |eta| bins; common ln pt range %.3f - %.3f (%.0f - %.0f GeV), %d points"
     % (J, sorted({m.K for m, _ in models}), len(models), x0, x1, np.exp(x0), np.exp(x1), a.npts),
     "# s_i / s_1 and, in brackets, the relative Frobenius error of truncating to chi = 1, 2, 3, 4, 6"]
for g in GAUGES:
    L.append("\n== gauge: %s" % g)
    for name, (s, err) in [(k, v) for k, v in res[g].items() if k != "per bin (n | x)"]:
        L.append("  %-30s %s   [%s]" % (name, " ".join("%.1e" % v for v in s[: a.nshow]),
                                        ", ".join("%.3f" % err[k - 1] for k in (1, 2, 3, 4, 6) if k <= len(err))))
    L.append("  per |eta| bin (n | x), truncation error at chi = 1, 2, 3, 4:")
    for lab, (s, err) in zip(labels, res[g]["per bin (n | x)"]):
        L.append("    %-9s %s" % (lab, "  ".join("%.3f" % err[k - 1] for k in (1, 2, 3, 4))))
print("\n".join(L))
with open(os.path.join(outdir, "tt_svd.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
names = [k for k in res["fit"] if k != "per bin (n | x)"]
fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4), sharey=True)
for ax, name in zip(axes, names):
    for g, st in zip(GAUGES, ("o-", "s--", "^:")):
        s = res[g][name][0]
        ax.semilogy(np.arange(1, len(s) + 1), s, st, ms=4, label=g)
    ax.set_title(name, fontsize=10); ax.set_xlabel("index i"); ax.grid(alpha=0.3)
axes[0].set_ylabel("$s_i / s_1$"); axes[0].legend(title="gauge", fontsize=8)
fig.suptitle("singular values of the unfoldings of c$_n$(ln $p_T$, |%s| bin)" % G.get("var", "eta"))
fig.tight_layout()
fig.savefig(os.path.join(outdir, "tt_svd.png"), dpi=120)
print("\noutputs in", outdir)
