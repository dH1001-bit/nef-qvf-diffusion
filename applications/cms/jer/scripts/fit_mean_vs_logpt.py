#!/usr/bin/env python3
"""Mean response against ln(pt_gen) in every |eta| bin, with weighted least-squares fits of
a 1st- and a 2nd-order polynomial in x = ln(pt_gen / --pt-ref).

Input: the histograms.npz (and binning.json next to it) written by plot_response_all_bins.py.
Nothing is re-read from the export.

    python3 scripts/fit_mean_vs_logpt.py --load fig/response_bins_receta/histograms.npz

Points: one per unmerged (pt_gen, |eta|) cell of binning.json; cells that were merged across
|eta| are skipped (their mean mixes |eta| regions).  Per point:
    x      = ln(<pt_gen>_w / pt_ref), <pt_gen>_w the weighted mean pt_gen of the cell
    mean   = sum w r / sum w over r in the histogram range (as in summary.txt)
    error  = sqrt(sum_k S2_k (r_k - mean)^2) / sum w,  S2_k = sum w^2 in fine response bin k:
             the standard error of a weighted mean, exact up to the 0.002 binning
Fit: minimise chi^2 = sum ((mean - P(x)) / error)^2 for P of degree 1 and 2.  Reported per
|eta| bin: coefficients with errors, chi^2/ndf, p-value and the chi^2 gain of degree 2 over 1
(Delta chi^2 ~ chi^2_1 if the quadratic term is zero and the linear model is right).

Outputs in --outdir (default: the directory of --load): mean_vs_logpt.png, mean_vs_logpt.txt,
mean_vs_logpt.json.
"""
import argparse, json, os
import numpy as np
from scipy.stats import chi2 as chi2_dist

ap = argparse.ArgumentParser()
ap.add_argument("--load", default="fig/response_bins_receta/histograms.npz",
                help="histograms.npz of plot_response_all_bins.py")
ap.add_argument("--binning", default=None, help="binning.json (default: next to --load)")
ap.add_argument("--min-neff", type=float, default=None,
                help="no binning.json: redo the merging with this threshold (as plot_response_all_bins.py)")
ap.add_argument("--pt-ref", type=float, default=100.0, help="x = ln(pt_gen / pt_ref) [GeV]")
ap.add_argument("--min-ndf", type=int, default=1,
                help="fit a degree only if points - (degree + 1) >= this (at least 1, for a chi^2 test)")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

outdir = a.outdir or os.path.dirname(os.path.abspath(a.load))
os.makedirs(outdir, exist_ok=True)

with np.load(a.load) as z:
    PT, ETA, R = z["pt_edges"], z["eta_edges"], z["r_edges"]
    var = str(z["var"])
    sw2_f = z["sw2"]
    W, W2, Wlo, Whi, S1, SG = (z[k] for k in ("W", "W2", "Wlo", "Whi", "S1", "SG"))
npt, neta = len(PT) - 1, len(ETA) - 1
rc = 0.5 * (R[1:] + R[:-1])

# --- which fine cells survived unmerged
unmerged = np.zeros((npt, neta), bool)
bpath = a.binning or os.path.join(os.path.dirname(os.path.abspath(a.load)), "binning.json")
if a.min_neff is None and os.path.exists(bpath):
    with open(bpath) as fh:
        edges_per_pt = json.load(fh)["eta_edges_per_pt"]
    for ipt, ed in enumerate(edges_per_pt):
        for lo, hi in zip(ed[:-1], ed[1:]):
            ie = int(np.argmin(np.abs(ETA - lo)))
            if ie + 1 <= neta and np.isclose(ETA[ie + 1], hi):    # the group is exactly one fine cell
                unmerged[ipt, ie] = True
    print("merging from", bpath)
else:
    if a.min_neff is None:
        raise SystemExit("no %s: pass --binning or --min-neff" % bpath)
    Wm, W2m = W.reshape(npt, neta), W2.reshape(npt, neta)
    for ipt in range(npt):                       # the greedy merge of plot_response_all_bins.py
        groups, start, gw, gw2 = [], 0, 0.0, 0.0
        for ie in range(neta):
            gw += Wm[ipt, ie]; gw2 += W2m[ipt, ie]
            if gw2 > 0 and gw * gw / gw2 >= a.min_neff:
                groups.append([start, ie]); start, gw, gw2 = ie + 1, 0.0, 0.0
        if start < neta:
            if groups:
                groups[-1][1] = neta - 1
            else:
                groups.append([start, neta - 1])
        for s, e in groups:
            if s == e:
                unmerged[ipt, s] = True
    print("merging redone with --min-neff %g" % a.min_neff)

# --- per fine cell: x, mean, error
with np.errstate(invalid="ignore", divide="ignore"):
    Win = (W - Wlo - Whi).reshape(npt, neta)
    mean = S1.reshape(npt, neta) / Win
    ptm = SG.reshape(npt, neta) / W.reshape(npt, neta)
    x = np.log(ptm / a.pt_ref)
    var_m = np.einsum("pek,pek->pe", sw2_f.reshape(npt, neta, -1),
                      (rc[None, None, :] - mean[..., None]) ** 2)
    err = np.sqrt(var_m) / Win
    neff = W.reshape(npt, neta) ** 2 / W2.reshape(npt, neta)

def wls(xv, yv, ev, deg):
    """weighted least squares y = sum_k b_k x^k: coefficients, covariance, chi^2."""
    A = np.vander(xv, deg + 1, increasing=True)
    Aw = A / ev[:, None]
    cov = np.linalg.inv(Aw.T @ Aw)
    b = cov @ (Aw.T @ (yv / ev))
    chi2 = float(np.sum(((yv - A @ b) / ev) ** 2))
    return b, cov, chi2

results, lines = [], []
for ie in range(neta):
    ok = unmerged[:, ie] & np.isfinite(mean[:, ie]) & np.isfinite(x[:, ie]) & (err[:, ie] > 0)
    xv, yv, ev = x[ok, ie], mean[ok, ie], err[ok, ie]
    d = dict(abs_eta=[float(ETA[ie]), float(ETA[ie + 1])], n_points=int(ok.sum()),
             pt_bins=[[float(PT[i]), float(PT[i + 1])] for i in np.flatnonzero(ok)],
             x=xv.tolist(), pt_mean=(a.pt_ref * np.exp(xv)).tolist(), mean=yv.tolist(), err=ev.tolist(),
             n_eff=neff[ok, ie].tolist(), fits={})
    for deg in (1, 2):
        ndf = len(xv) - deg - 1
        if ndf < max(1, a.min_ndf):
            continue
        b, cov, c2 = wls(xv, yv, ev, deg)
        d["fits"][str(deg)] = dict(coef=b.tolist(), err=np.sqrt(np.diag(cov)).tolist(), cov=cov.tolist(),
                                   chi2=c2, ndf=ndf, p=float(chi2_dist.sf(c2, ndf)))
    f = d["fits"]
    if "1" in f and "2" in f:
        dc = f["1"]["chi2"] - f["2"]["chi2"]
        d["dchi2_2_vs_1"] = dc
        d["p_dchi2"] = float(chi2_dist.sf(dc, 1))
    results.append(d)

# --- text output
hdr = ("|%s| %-9s %3s  %-38s %10s %7s   %-56s %10s %7s %8s %7s"
       % (var, "", "pts", "deg 1: b0, b1", "chi2/ndf", "p", "deg 2: b0, b1, b2", "chi2/ndf", "p", "dchi2", "p(dc)"))
lines.append("# mean response vs x = ln(<pt_gen> / %g GeV); merged |%s| cells skipped" % (a.pt_ref, var))
lines.append("# P(x) = b0 + b1 x [+ b2 x^2]; errors from the weighted least-squares covariance")
lines.append("# dchi2 = chi2(deg 1) - chi2(deg 2): ~chi2_1 if b2 = 0; small p(dc) means the curvature is needed")
lines.append(hdr)
for d in results:
    s = "%4.2f-%-4.2f %4s %3d  " % (d["abs_eta"][0], d["abs_eta"][1], "", d["n_points"])
    for deg, w in (("1", 38), ("2", 56)):
        if deg in d["fits"]:
            g = d["fits"][deg]
            co = ", ".join("%.4f(%.0f)" % (b, 1e4 * e) for b, e in zip(g["coef"], g["err"]))
            s += "%-*s %10s %7.3f   " % (w, co, "%.1f/%d" % (g["chi2"], g["ndf"]), g["p"])
        else:
            s += "%-*s %10s %7s   " % (w, "(too few points)", "", "")
    if "dchi2_2_vs_1" in d:
        s += "%8.2f %7.3f" % (d["dchi2_2_vs_1"], d["p_dchi2"])
    lines.append(s.rstrip())
lines.append("# coefficient(error): the error in units of 1e-4")
print("\n".join(lines))
with open(os.path.join(outdir, "mean_vs_logpt.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
with open(os.path.join(outdir, "mean_vs_logpt.json"), "w") as fh:
    json.dump(dict(var=var, pt_ref=a.pt_ref, eta_bins=results), fh, indent=1)

# --- plot: one column per |eta| bin, data + fits on top, pulls below
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ncol = min(neta, 4)
nrow = int(np.ceil(neta / ncol))
fig = plt.figure(figsize=(4.2 * ncol, 5.0 * nrow))
outer = fig.add_gridspec(nrow, ncol, hspace=0.45, wspace=0.28)   # room for the GeV axis of the next row
colors = {"1": "C0", "2": "C3"}
for k, d in enumerate(results):
    r_, c_ = divmod(k, ncol)
    inner = outer[r_, c_].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(inner[0])
    axp = fig.add_subplot(inner[1], sharex=ax)
    xv, yv, ev = map(np.asarray, (d["x"], d["mean"], d["err"]))
    ax.set_title("%.2f < |%s| < %.2f" % (d["abs_eta"][0], var, d["abs_eta"][1]), fontsize=10)
    if len(xv):
        ax.errorbar(xv, yv, ev, fmt="ko", ms=3.5, capsize=2, label="mean")
        xx = np.linspace(xv.min() - 0.1, xv.max() + 0.1, 200)
        for deg, g in d["fits"].items():
            b = np.asarray(g["coef"])
            ax.plot(xx, np.polyval(b[::-1], xx), color=colors[deg], lw=1.4,
                    label="deg %s: $\\chi^2$/ndf %.1f/%d" % (deg, g["chi2"], g["ndf"]))
            axp.plot(xv + (0.02 if deg == "2" else -0.02), (yv - np.polyval(b[::-1], xv)) / ev, "o",
                     color=colors[deg], ms=3.5)
    else:
        ax.text(0.5, 0.5, "no unmerged cells", transform=ax.transAxes, ha="center")
    axp.axhline(0, color="gray", lw=0.8)
    for v in (-2, 2):
        axp.axhline(v, color="gray", lw=0.6, ls=":")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3); axp.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), visible=False)
    axp.set_xlabel("ln($p_T^{gen}$ / %g GeV)" % a.pt_ref)
    if c_ == 0:
        ax.set_ylabel("mean response")
        axp.set_ylabel("pull")
    # secondary axis in GeV
    sec = ax.secondary_xaxis("top", functions=(lambda t: a.pt_ref * np.exp(t),
                                               lambda p: np.log(np.maximum(p, 1e-9) / a.pt_ref)))
    sec.set_xticks([100, 200, 500, 1000, 2000])
    sec.tick_params(labelsize=7)
fig.suptitle("mean response vs ln $p_T^{gen}$ (merged |%s| cells skipped)" % var, y=0.995)
png = os.path.join(outdir, "mean_vs_logpt.png")
fig.savefig(png, dpi=130, bbox_inches="tight")
print("\n%s, mean_vs_logpt.txt and mean_vs_logpt.json in %s" % (os.path.basename(png), outdir))
