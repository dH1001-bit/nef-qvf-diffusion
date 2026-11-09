#!/usr/bin/env python3
"""Amplitude-fit coefficients against ln(pt_gen), one figure per |eta| bin: a first look at
whether the shape changes smoothly enough with pt_gen for a global (pt-smooth) fit.

Input: fits.json and histograms.npz of plot_response_all_bins.py (same directory by default).

    python3 scripts/plot_coefficients_vs_logpt.py --fits fig/response_bins_receta/fits.json

Two views of every cell's Born model p(r) = N(r; mu_ref, sigma_ref) |sum_n c_n phi_n(z)|^2,
z = (r - mu_ref) / sigma_ref:

1. coefficients_eta<i>.png: Re c_n, Im c_n and |c_n|, n = 0 ... J, global phase fixed by
   c_0 real >= 0 (as stored in fits.json).  These are NOT unique: replacing any root of
   h(z) = sum c_n phi_n(z) by its complex conjugate leaves |h|^2 on the real axis unchanged,
   and in --amp-ref free-all mode (mu_ref, sigma_ref) trade off against c_1, c_2.  Jumps from
   one pt_gen bin to the next can therefore be gauge artefacts, not physics.

2. moments_eta<i>.png: the gauge-invariant view.  With m, s the mean and standard deviation
   of the fitted model itself and w = (r - m) / s,
       a_k = E_p[phi_k(w)] = E_p[He_k(w)] / sqrt(k!),       k = 3 ... 2J
   (a_0 = 1, a_1 = a_2 = 0 by construction).  These are the standardised Hermite
   (Gram-Charlier) moments: a_3 = skewness / sqrt(6), a_4 = excess kurtosis / sqrt(24).
   They depend only on the density p, so they do not see the phase, the root flips or the
   reference trade-off.  The figure also shows m and s.  With --moments ref, the a_k are taken
   in the reference variable z instead (a_1, a_2 then nonzero, but no longer invariant).
   E_p is computed exactly by Gauss-Hermite quadrature: the integrand is N(z) times a
   polynomial of degree <= 4J.

   Caveat: phi_k(w) grows like w^k, so a_k for large k measures the far tails of the model,
   where it extrapolates beyond the data; these values are large and erratic even for good
   fits.  Only k <= --kmax (default 8) is drawn; the json keeps all of them.

3. quantiles_eta<i>.png: a bounded invariant view of the same density.  Median Q_50, the
   half-width sigma_68 = (Q_84 - Q_16)/2 relative to the median, and the standardised quantiles
   (Q_p - Q_50)/sigma_68 for p = 0.1, 1, 5, 95, 99, 99.9 %; the dashed line is the Gaussian
   value.  Computed from the model's CDF on a fine grid.

Only unmerged cells are used (a merged cell mixes |eta| regions).  x = ln(<pt_gen>_w / --pt-ref)
with the cell's weighted mean pt_gen.  Open markers: the optimiser did not report convergence.
The values also go to coefficients_vs_logpt.json.
"""
import argparse, json, os
from math import factorial
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fits", default="fig/response_bins_receta/fits.json", help="fits.json of plot_response_all_bins.py")
ap.add_argument("--load", default=None, help="histograms.npz (default: next to --fits)")
ap.add_argument("--pt-ref", type=float, default=100.0, help="x = ln(pt_gen / pt_ref) [GeV]")
ap.add_argument("--moments", default="std", choices=("std", "ref"),
                help="a_k in w = (r - m)/s of the model (std, invariant) or in the reference z (ref)")
ap.add_argument("--kmax", type=int, default=8, help="highest moment a_k drawn (0: all, up to 2J)")
ap.add_argument("--outdir", default=None, help="default: <directory of --fits>/coefficients")
a = ap.parse_args()

fdir = os.path.dirname(os.path.abspath(a.fits))
outdir = a.outdir or os.path.join(fdir, "coefficients")
os.makedirs(outdir, exist_ok=True)

with open(a.fits) as fh:
    F = json.load(fh)
var = F.get("var", "receta")
with np.load(a.load or os.path.join(fdir, "histograms.npz")) as z:
    PT, ETA = z["pt_edges"], z["eta_edges"]
    W, SG = z["W"], z["SG"]
npt, neta = len(PT) - 1, len(ETA) - 1
with np.errstate(invalid="ignore", divide="ignore"):
    ptmean = (SG / W).reshape(npt, neta)

def cell_index(edges, value):
    i = int(np.argmin(np.abs(edges - value)))
    return i if np.isclose(edges[i], value) else None

def amplitude_of(d):
    """(c, mu_ref, sigma_ref, converged) from one fits.json entry; also the older flat format."""
    am = d.get("amplitude")
    if am is not None:
        return (np.asarray(am["c_re"]) + 1j * np.asarray(am["c_im"]), am["mu_ref"], am["sigma_ref"],
                am.get("converged", True))
    if "c_re" in d:                                   # before --amp-ref: reference = the Gaussian MLE
        return np.asarray(d["c_re"]) + 1j * np.asarray(d["c_im"]), d["mu"], d["sigma"], d.get("converged", True)
    return None

# --- Gauss-Hermite (probabilists') quadrature: sum_i g_i f(t_i) = E_{N(0,1)}[f] exactly for deg f <= 2 nq - 1
J_all = [len(amplitude_of(d)[0]) - 1 for d in F["fits"] if amplitude_of(d) is not None]
if not J_all:
    raise SystemExit("no amplitude fits in " + a.fits)
J = max(J_all)
t, g = np.polynomial.hermite_e.hermegauss(2 * J + 2)
g = g / np.sqrt(2 * np.pi)
K = 2 * J
sqf = np.sqrt([float(factorial(k)) for k in range(K + 1)])

def phi(zv, n):
    return np.polynomial.hermite_e.hermevander(zv, n) / sqf[: n + 1]

def model_moments(c, mu_ref, sig_ref):
    """mean m, sd s (in r) and a_k, k = 0 ... 2J, of N(z) |sum c_n phi_n(z)|^2."""
    q = np.abs(phi(t, len(c) - 1) @ c) ** 2          # |h|^2 at the nodes
    norm = g @ q                                      # = ||c||^2 = 1 up to rounding
    zm = (g @ (q * t)) / norm
    zs = np.sqrt((g @ (q * t * t)) / norm - zm ** 2)
    m, s = mu_ref + sig_ref * zm, sig_ref * zs
    w = (t - zm) / zs if a.moments == "std" else t
    ak = (g * q) @ phi(w, K) / norm
    return m, s, ak, norm

PQ = np.array([0.001, 0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99, 0.999])
ZG = np.array([-3.0902, -2.3263, -1.6449, -0.9945, 0.0, 0.9945, 1.6449, 2.3263, 3.0902])  # Gaussian, /sigma_68

def model_quantiles(c, mu_ref, sig_ref):
    """quantiles PQ of the model in r, from its CDF on a fine grid (+-14 sigma_ref)."""
    zg = np.linspace(-14, 14, 28001)
    pz = np.exp(-0.5 * zg * zg) * np.abs(phi(zg, len(c) - 1) @ c) ** 2
    cdf = np.r_[0.0, np.cumsum(0.5 * (pz[1:] + pz[:-1]))]
    cdf /= cdf[-1]
    return mu_ref + sig_ref * np.interp(PQ, cdf, zg)

rows = {ie: [] for ie in range(neta)}
skipped = 0
for d in F["fits"]:
    amp = amplitude_of(d)
    if amp is None:
        continue
    ip, ie0, ie1 = cell_index(PT, d["pt_gen"][0]), cell_index(ETA, d["abs_eta"][0]), cell_index(ETA, d["abs_eta"][1])
    if ip is None or ie0 is None or ie1 is None or ie1 != ie0 + 1:
        skipped += 1                                   # merged across |eta|
        continue
    c, mr, sr, ok = amp
    m, s, ak, norm = model_moments(c, mr, sr)
    rows[ie0].append(dict(pt_gen=d["pt_gen"], pt_mean=float(ptmean[ip, ie0]),
                          x=float(np.log(ptmean[ip, ie0] / a.pt_ref)), n_eff=d.get("n_eff"),
                          converged=bool(ok), mu_ref=mr, sigma_ref=sr, c_re=c.real.tolist(), c_im=c.imag.tolist(),
                          model_mean=float(m), model_sd=float(s), a=ak.tolist(), norm=float(norm),
                          quantiles=dict(zip(("%g" % p for p in PQ), model_quantiles(c, mr, sr).tolist()))))
print("%d cells, %d merged cells skipped, J = %d, moments in %s variable"
      % (sum(map(len, rows.values())), skipped, J, "standardised" if a.moments == "std" else "reference"))
worst = max((abs(r["norm"] - 1) for v in rows.values() for r in v), default=0)
print("max | ||c||^2 - 1 | from the quadrature: %.1e" % worst)

with open(os.path.join(outdir, "coefficients_vs_logpt.json"), "w") as fh:
    json.dump(dict(var=var, pt_ref=a.pt_ref, moments=a.moments,
                   eta_bins=[dict(abs_eta=[float(ETA[ie]), float(ETA[ie + 1])], cells=rows[ie]) for ie in range(neta)]),
              fh, indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def grid(n):
    ncol = min(n, 4 if n <= 12 else 5)
    return int(np.ceil(n / ncol)), ncol

def points(ax, x, y, ok, **kw):
    """line through all points, filled markers for converged fits, open for the rest."""
    ax.plot(x, y, "-", lw=1.0, alpha=0.7, color=kw["color"])
    ax.plot(x[ok], y[ok], "o", ms=3.5, color=kw["color"], label=kw.get("label"))
    ax.plot(x[~ok], y[~ok], "o", ms=4.5, mfc="none", color=kw["color"])

def finish(fig, axes, n_used, title, path):
    for ax in axes.flat[n_used:]:
        ax.set_visible(False)
    for ax in axes.flat[:n_used]:
        ax.axhline(0, color="gray", lw=0.6)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("ln($p_T^{gen}$ / %g GeV)" % a.pt_ref, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)

n_png = 0
for ie in range(neta):
    cells = sorted(rows[ie], key=lambda r: r["x"])
    if not cells:
        continue
    lab = "%.2f < |%s| < %.2f" % (ETA[ie], var, ETA[ie + 1])
    x = np.array([r["x"] for r in cells])
    ok = np.array([r["converged"] for r in cells])
    Jc = min(len(r["c_re"]) for r in cells) - 1
    C = np.array([np.asarray(r["c_re"][: Jc + 1]) + 1j * np.asarray(r["c_im"][: Jc + 1]) for r in cells])

    # 1. phase-fixed c_n
    nr, nc_ = grid(Jc + 1)
    fig, axes = plt.subplots(nr, nc_, figsize=(3.3 * nc_, 2.6 * nr), sharex=True, squeeze=False)
    for n in range(Jc + 1):
        ax = axes.flat[n]
        points(ax, x, C[:, n].real, ok, color="C0", label="Re")
        points(ax, x, C[:, n].imag, ok, color="C3", label="Im")
        ax.plot(x, np.abs(C[:, n]), "--", color="gray", lw=1.0, label="|c|")
        ax.set_title("$c_{%d}$" % n, fontsize=10)
        if n == 0:
            ax.legend(fontsize=7)
    finish(fig, axes, Jc + 1, "amplitude coefficients (phase: $c_0$ real), %s" % lab,
           os.path.join(outdir, "coefficients_eta%d.png" % ie))

    # 2. invariant moments (+ model mean and sd)
    A = np.array([r["a"] for r in cells])
    kmax = 2 * Jc if a.kmax <= 0 else min(a.kmax, 2 * Jc)
    ks = list(range(3 if a.moments == "std" else 1, kmax + 1))
    nr, nc_ = grid(len(ks) + 2)
    fig, axes = plt.subplots(nr, nc_, figsize=(3.3 * nc_, 2.6 * nr), sharex=True, squeeze=False)
    points(axes.flat[0], x, np.array([r["model_mean"] for r in cells]), ok, color="k")
    axes.flat[0].set_title("model mean $m$", fontsize=10)
    points(axes.flat[1], x, np.array([r["model_sd"] for r in cells]) / np.array([r["model_mean"] for r in cells]),
           ok, color="k")
    axes.flat[1].set_title("model sd / mean  $s/m$", fontsize=10)
    for j, k in enumerate(ks):
        ax = axes.flat[j + 2]
        points(ax, x, A[:, k], ok, color="C2")
        extra = {3: "  (skew/$\\sqrt{6}$)", 4: "  (exc. kurt/$\\sqrt{24}$)"}.get(k, "") if a.moments == "std" else ""
        ax.set_title("$a_{%d}$%s" % (k, extra), fontsize=10)
    finish(fig, axes, len(ks) + 2,
           "Hermite moments $a_k = E[\\phi_k(w)]$, $w$ = %s, %s"
           % ("$(r-m)/s$" if a.moments == "std" else "$(r-\\mu_{ref})/\\sigma_{ref}$", lab),
           os.path.join(outdir, "moments_eta%d.png" % ie))

    # 3. standardised quantiles
    Q = np.array([[r["quantiles"]["%g" % p] for p in PQ] for r in cells])
    med, s68 = Q[:, 4], 0.5 * (Q[:, 5] - Q[:, 3])
    tails = [0, 1, 2, 6, 7, 8]
    nr, nc_ = grid(len(tails) + 2)
    fig, axes = plt.subplots(nr, nc_, figsize=(3.3 * nc_, 2.6 * nr), sharex=True, squeeze=False)
    points(axes.flat[0], x, med, ok, color="k")
    axes.flat[0].set_title("median $Q_{50}$", fontsize=10)
    points(axes.flat[1], x, s68 / med, ok, color="k")
    axes.flat[1].set_title("$\\sigma_{68}$ / median", fontsize=10)
    for j, i in enumerate(tails):
        ax = axes.flat[j + 2]
        points(ax, x, (Q[:, i] - med) / s68, ok, color="C4")
        ax.axhline(ZG[i] / 0.9945, color="C4", ls="--", lw=0.9)
        ax.set_title("$(Q_{%g} - Q_{50})/\\sigma_{68}$" % (100 * PQ[i]), fontsize=10)
    finish(fig, axes, len(tails) + 2, "model quantiles (dashed: Gaussian), %s" % lab,
           os.path.join(outdir, "quantiles_eta%d.png" % ie))
    n_png += 3
print("%d figures and coefficients_vs_logpt.json in %s" % (n_png, outdir))
