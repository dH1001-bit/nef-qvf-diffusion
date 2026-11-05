#!/usr/bin/env python3
"""The weighted jet response r = pt_rec / pt_gen in one pt_gen and eta window,
read straight from the campaign export (all 72 files, every jet), plotted on
a linear and a log scale.  numpy + matplotlib only; written top to bottom so
it can be stepped through in ipython.

    python3 scripts/plot_response.py --pt 300 500 --eta 0 0.5
    python3 scripts/plot_response.py --pt 1000 1400 --eta 1.3 2.5 --var geneta
    ipython -i scripts/plot_response.py -- --pt 300 500 --eta 0 0.5 --max-files 9

The window is in GENERATOR pt: binning in reconstructed pt would select jets
by the very fluctuation being measured and bias the response.  The eta window
is on |var|, by default the reconstructed |eta| (the JER convention); recy,
geneta and geny are also stored.

No trigger is needed here.  This is simulation: every event enters with its
cross-section weight, and the nine H_T slices add up to the inclusive QCD
spectrum.  What plays the role of a trigger is the H_T slicing, and the
per-slice table printed below shows which slices populate the window.  All
slices must be read: dropping one leaves a hole in the spectrum, not a
smaller sample.
"""
import argparse, glob, os, re, time
import numpy as np
import sys
from scipy.optimize import minimize
# --------------------------------------------------------------------------
# 0. imports: the nefqvf package (repository src/) and the jerfit package
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
APP = os.path.abspath(os.path.join(HERE, ".."))                  # applications/cms/jer
REPO = os.path.abspath(os.path.join(APP, "..", "..", ".."))        # the repository root
for p in (APP, os.path.join(REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from nefqvf import Normal, NormalParams, product_matrices, ratio_coefficients   # noqa: E402
from jerfit import load, select, fit_bin, fit_core    

ap = argparse.ArgumentParser()
ap.add_argument("--pt", nargs=2, type=float, default=(300.0, 500.0), metavar=("LO", "HI"),
                help="pt_gen window [GeV]")
ap.add_argument("--eta", nargs=2, type=float, default=(0.0, 0.5), metavar=("LO", "HI"),
                help="window in |var|")
ap.add_argument("--var", default="receta", choices=("receta", "recy", "geneta", "geny"))
ap.add_argument("--max-dr", type=float, default=0.4, help="matching radius, at most the export's 0.4")
ap.add_argument("--range", nargs=2, type=float, default=(0.0, 2.0), metavar=("LO", "HI"),
                help="response range of the histogram")
ap.add_argument("--bins", type=int, default=100)
ap.add_argument("--files", default="/scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz")
ap.add_argument("--max-files", type=int, default=None, help="only the first N files (a quick look; biased spectrum)")
ap.add_argument("--out", default=None, help="figure path (default: fig/response_pt<lo>-<hi>_<var><lo>-<hi>.png)")
a = ap.parse_args()

files = sorted(glob.glob(a.files))
if a.max_files:
    files = files[: a.max_files]
if not files:
    raise SystemExit("no files match " + a.files)

# --- read: genpt and the eta column first, everything else only for selected rows
r_all, w_all = [], []
per_slice = {}
n_read = 0
t0 = time.time()
for i, f in enumerate(files):
    with np.load(f) as z:
        genpt = z["genpt"]
        eta = np.abs(z[a.var])
        sel = (genpt >= a.pt[0]) & (genpt < a.pt[1]) & (eta >= a.eta[0]) & (eta < a.eta[1])
        if a.max_dr < 0.4:
            sel &= z["dR"] < a.max_dr
        n_read += len(genpt)
        idx = np.flatnonzero(sel)
        r = z["recpt"][idx].astype(np.float64) / genpt[idx]
        w = z["weight"][idx].astype(np.float64)
        sl = str(z["slice"])
    r_all.append(r); w_all.append(w)
    s = per_slice.setdefault(sl, [0, 0.0, 0.0])
    s[0] += len(w); s[1] += w.sum(); s[2] += (w * w).sum()
    print("  %2d/%d %-32s %9d selected  %4.0f s" % (i + 1, len(files), os.path.basename(f), len(w), time.time() - t0),
          flush=True)

r = np.concatenate(r_all)
w = np.concatenate(w_all)
n_eff = w.sum() ** 2 / (w * w).sum() if len(w) else 0.0
print("\n%d rows read, %d in the window  %g <= pt_gen < %g GeV, %g <= |%s| < %g;  sum w = %.4g pb, n_eff = %.0f"
      % (n_read, len(r), a.pt[0], a.pt[1], a.eta[0], a.var, a.eta[1], w.sum(), n_eff))
if len(r) == 0:
    raise SystemExit("empty window")

# --- which H_T slices populate the window (the simulation's 'trigger')
def order(sl):
    m = re.match(r"(\d+)", sl); return int(m.group(1)) if m else 0
print("\nH_T slice        rows   share of sum w   n_eff")
for sl in sorted(per_slice, key=order):
    n, sw, sw2 = per_slice[sl]
    if n:
        print("  %-12s %9d   %8.4f       %9.0f" % (sl, n, sw / w.sum(), sw * sw / sw2))

# --- summary numbers: weighted mean and rms inside the histogram range, and the overflow
inside = (r >= a.range[0]) & (r < a.range[1])
mean = np.average(r[inside], weights=w[inside])
rms = np.sqrt(np.average((r[inside] - mean) ** 2, weights=w[inside]))
med = np.interp(0.5, np.cumsum(w[np.argsort(r)]) / w.sum(), np.sort(r))
print("\nweighted mean %.4f  rms %.4f  median %.4f   (inside %g <= r < %g; %.2g%% of sum w outside)"
      % (mean, rms, med, *a.range, 100 * w[~inside].sum() / w.sum()))

# --- the histogram: density in r, errors from sum w^2 per bin
edges = np.linspace(*a.range, a.bins + 1)
sw, _ = np.histogram(r, edges, weights=w)
sw2, _ = np.histogram(r, edges, weights=w * w)
cnt, _ = np.histogram(r, edges)
norm = w.sum() * np.diff(edges)
dens, err = sw / norm, np.sqrt(sw2) / norm
centres = 0.5 * (edges[1:] + edges[:-1])
gauss = np.exp(-0.5 * ((centres - mean) / rms) ** 2) / (np.sqrt(2 * np.pi) * rms)


# amplitude fit
def weighted_moments(r, w, params, k_max):
    """<phi_k>_w and its error for k = 0 ... k_max."""
    B = Normal.basis(r, k_max, params)
    mean = (w @ B) / w.sum()
    err = np.sqrt(((w[:, None] ** 2) * (B - mean) ** 2).sum(axis=0)) / w.sum()
    return mean, err

J = 6
c_amp = np.zeros((J+1), dtype = complex)
c_amp[0] = 1
c_amp = c_amp + 0.01j
w_matrix = np.eye(2*J+1)

g_baselaw = NormalParams(mean, rms)
g_prob = Normal.prob(centres, g_baselaw)

R_c = np.zeros((2*J+1))
r_moments, r_errors = weighted_moments(r, w, g_baselaw, 2*J)

Phi_g = product_matrices(Normal, g_baselaw, J)               # (2K+1, K+1, K+1)

def loss_function(c):
    c = c / np.linalg.norm(c)
    c_tmp = c.reshape((-1,1))
    for i in range(2*J+1):
        R_c[i] = c_tmp.conj().T @ Phi_g[i] @ c_tmp
    quad_matrix = R_c - r_moments
    J_c = 0.5*(quad_matrix.T @ w_matrix @ quad_matrix)
    return J_c

def logloss(c, mu, sigma, r, w):
    c = np.asarray(c, dtype=complex)
    c = c / np.linalg.norm(c)
    params = NormalParams(mu, sigma)
    h = Normal.basis_dot(r, c, params)
    with np.errstate(divide="ignore"):                 # h = 0 at a root of the amplitude: log p = -inf there
        return -np.sum(w * (Normal.log_prob(r, params) + 2.0 * np.log(np.abs(h))))

c_real_expended = np.concatenate((np.real(c_amp), np.imag(c_amp)), axis=None)

optimizer_result = minimize(logloss, c_real_expended, args=(mean, rms, r, w))
#optimizer_result = minimize(loss_function, c_real_expended)

print(optimizer_result)

c_real_expended = optimizer_result.x/np.linalg.norm(optimizer_result.x)
c_amp = c_real_expended[:J+1] + 1j*c_real_expended[J+1:]

print(c_amp)

amplitude_approx = g_prob*np.abs(Normal.basis_dot(centres, c_amp, g_baselaw))**2


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, log in zip(axes, (False, True)):
    ok = cnt > 0
    ax.errorbar(centres[ok], dens[ok], yerr=err[ok], fmt=".", ms=3, color="k", lw=0.8, label="jets (weighted)")
    ax.plot(centres, gauss, color="C0", lw=1, label="Gaussian, mean %.3f rms %.3f" % (mean, rms))
    ax.set_xlabel(r"$p_T^{\rm rec}/p_T^{\rm gen}$")
    ax.set_ylabel("1/N dN/dr")
    ax.plot(centres, amplitude_approx, label=r"amplitude fit" )
    if log:
        ax.set_yscale("log"); ax.set_ylim(max(dens[ok].min() / 3, 1e-7), dens.max() * 3)
    ax.legend(fontsize=8)
fig.suptitle(r"UL18 MadGraph, $%g \leq p_T^{\rm gen} < %g$ GeV, $%g \leq |%s| < %g$;  %d jets, $n_{\rm eff}$ = %.0f"
             % (a.pt[0], a.pt[1], a.eta[0], a.var, a.eta[1], len(r), n_eff), fontsize=10)
fig.tight_layout()
out = a.out or os.path.join("fig", "response_pt%g-%g_%s%g-%g.png" % (*a.pt, a.var, *a.eta))
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.savefig(out, dpi=150)
print("figure:", out)
