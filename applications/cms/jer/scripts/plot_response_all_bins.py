#!/usr/bin/env python3
"""The weighted jet response r = pt_rec / pt_gen in EVERY cell of a pt_gen x |eta|
grid, read straight from the campaign export (all 72 files, every jet) in a
single pass, and plotted cell by cell (linear and log scale) into one directory.
numpy + matplotlib only; written top to bottom so it can be stepped through in
ipython.  plot_response.py does the same for one hand-picked window.

    python3 scripts/plot_response_all_bins.py
    python3 scripts/plot_response_all_bins.py --var geneta --outdir fig/resp_geneta
    python3 scripts/plot_response_all_bins.py --eta-edges 0 0.522 0.783 1.131 1.305 1.740 1.930 2.043 2.322 2.5 2.853 2.964 3.139 5.191
    python3 scripts/plot_response_all_bins.py --load fig/response_bins_receta/histograms.npz --bins 50   # replot, no reading
    ipython -i scripts/plot_response_all_bins.py -- --max-files 9

Default grid: the pt_gen and |y| edges of count_jets.py.  The last argument
example is the CMS JER scale-factor |eta| binning.

How it works: the files are read once.  Each jet gets a cell index
(ipt, ieta) and a fine response-bin index, and np.bincount adds w, w^2 and 1
into per-cell histograms.  Memory is one file plus the histograms, never the
jets themselves.  The fine histograms (--fine sub-bins per plotted bin) are
saved to <outdir>/histograms.npz, so --load can replot with another --bins
(any divisor of the stored bin count) without the ~10 min read.

Mean and rms are exact (from sum w r and sum w r^2 over the jets inside
--range); the median is interpolated in the fine histogram.

As in plot_response.py: the grid is in GENERATOR pt (binning in reconstructed
pt selects on the fluctuation being measured), no trigger is needed (the H_T
slices with their cross-section weights add up to the inclusive spectrum),
and every slice must be read.  summary.txt lists, per cell, which slice
dominates the summed weight.
"""
import argparse, glob, os, re, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pt-edges", nargs="+", type=float,
                default=[97, 140, 200, 280, 400, 550, 750, 1000, 1400, 2000, 3500, 1e5],
                help="pt_gen bin edges [GeV]")
ap.add_argument("--eta-edges", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
                help="bin edges in |var|")
ap.add_argument("--var", default="receta", choices=("receta", "recy", "geneta", "geny"))
ap.add_argument("--max-dr", type=float, default=0.4, help="matching radius, at most the export's 0.4")
ap.add_argument("--range", nargs=2, type=float, default=(0.0, 2.0), metavar=("LO", "HI"),
                help="response range of the histograms")
ap.add_argument("--bins", type=int, default=100, help="response bins in the plots")
ap.add_argument("--fine", type=int, default=10, help="stored sub-bins per plotted bin (median, rebinning)")
ap.add_argument("--min-rows", type=int, default=1, help="skip cells with fewer jets")
ap.add_argument("--files", default="/scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz")
ap.add_argument("--max-files", type=int, default=None, help="only the first N files (a quick look; biased spectrum)")
ap.add_argument("--outdir", default=None, help="output directory (default: fig/response_bins_<var>)")
ap.add_argument("--load", default=None, help="histograms.npz of an earlier run: replot without reading the export")
a = ap.parse_args()

outdir = a.outdir or os.path.join("fig", "response_bins_%s" % a.var)
os.makedirs(outdir, exist_ok=True)

if a.load:
    # --- replot: everything comes from the saved histograms
    with np.load(a.load) as z:
        PT, ETA, R = z["pt_edges"], z["eta_edges"], z["r_edges"]
        var = str(z["var"])
        sw_f, sw2_f, cnt_f = z["sw"], z["sw2"], z["cnt"]
        n, W, W2, Wlo, Whi, S1, S2, SG = (z[k] for k in ("n", "W", "W2", "Wlo", "Whi", "S1", "S2", "SG"))
        slice_names, slice_tab = list(z["slice_names"]), z["slice_tab"]
    npt, neta, nfine = len(PT) - 1, len(ETA) - 1, len(R) - 1
    nb = npt * neta
    if nfine % a.bins:
        raise SystemExit("--bins %d does not divide the %d stored response bins" % (a.bins, nfine))
    print("loaded %s: %d x %d cells, %d response bins in [%g, %g)" % (a.load, npt, neta, nfine, R[0], R[-1]))
else:
    PT, ETA = np.asarray(a.pt_edges), np.asarray(a.eta_edges)
    var = a.var
    npt, neta = len(PT) - 1, len(ETA) - 1
    nb = npt * neta                       # cell b = ipt * neta + ieta
    nfine = a.bins * a.fine
    R = np.linspace(*a.range, nfine + 1)

    files = sorted(glob.glob(a.files))
    if a.max_files:
        files = files[: a.max_files]
    if not files:
        raise SystemExit("no files match " + a.files)

    # --- accumulators: per cell (nb) and per cell x fine response bin (nb, nfine)
    sw_f, sw2_f, cnt_f = np.zeros((nb, nfine)), np.zeros((nb, nfine)), np.zeros((nb, nfine))
    n = np.zeros(nb, np.int64)
    W, W2 = np.zeros(nb), np.zeros(nb)            # all jets of the cell
    Wlo, Whi = np.zeros(nb), np.zeros(nb)         # weight below / above --range
    S1, S2 = np.zeros(nb), np.zeros(nb)           # sum w r, sum w r^2 inside --range
    SG = np.zeros(nb)                             # sum w pt_gen: the cell's mean pt for the summary plot
    per_slice = {}                                # slice -> (3, nb): rows, sum w, sum w^2

    def bc(idx, wt=None, size=nb):
        return np.bincount(idx, weights=wt, minlength=size)

    n_read = 0
    t0 = time.time()
    for i, f in enumerate(files):
        with np.load(f) as z:
            genpt = z["genpt"]
            ip = np.digitize(genpt, PT) - 1
            ie = np.digitize(np.abs(z[var]), ETA) - 1
            sel = (ip >= 0) & (ip < npt) & (ie >= 0) & (ie < neta)
            if a.max_dr < 0.4:
                sel &= z["dR"] < a.max_dr
            n_read += len(genpt)
            idx = np.flatnonzero(sel)
            b = ip[idx] * neta + ie[idx]
            g = genpt[idx].astype(np.float64)
            r = z["recpt"][idx].astype(np.float64) / g
            w = z["weight"][idx].astype(np.float64)
            sl = str(z["slice"])
        n += bc(b); W += bc(b, w); W2 += bc(b, w * w); SG += bc(b, w * g)
        ir = np.digitize(r, R) - 1
        lo, hi = ir < 0, ir >= nfine
        ins = ~(lo | hi)
        Wlo += bc(b[lo], w[lo]); Whi += bc(b[hi], w[hi])
        bi, ri, wi = b[ins], r[ins], w[ins]
        S1 += bc(bi, wi * ri); S2 += bc(bi, wi * ri * ri)
        k = bi * nfine + ir[ins]
        sw_f += bc(k, wi, nb * nfine).reshape(nb, nfine)
        sw2_f += bc(k, wi * wi, nb * nfine).reshape(nb, nfine)
        cnt_f += bc(k, None, nb * nfine).reshape(nb, nfine)
        s = per_slice.setdefault(sl, np.zeros((3, nb)))
        s[0] += bc(b); s[1] += bc(b, w); s[2] += bc(b, w * w)
        print("  %2d/%d %-32s %9d in the grid  %4.0f s" % (i + 1, len(files), os.path.basename(f), len(w),
                                                          time.time() - t0), flush=True)
    print("\n%d rows read, %d in the grid" % (n_read, n.sum()))

    def order(sl):
        m = re.match(r"(\d+)", sl); return int(m.group(1)) if m else 0
    slice_names = sorted(per_slice, key=order)
    slice_tab = np.array([per_slice[s] for s in slice_names]).reshape(len(slice_names), 3, nb)

    np.savez(os.path.join(outdir, "histograms.npz"), pt_edges=PT, eta_edges=ETA, r_edges=R, var=var,
             sw=sw_f, sw2=sw2_f, cnt=cnt_f, n=n, W=W, W2=W2, Wlo=Wlo, Whi=Whi, S1=S1, S2=S2, SG=SG,
             slice_names=np.array(slice_names), slice_tab=slice_tab)
    print("histograms:", os.path.join(outdir, "histograms.npz"))

# --- per-cell summary numbers
with np.errstate(invalid="ignore", divide="ignore"):
    Win = W - Wlo - Whi
    mean = S1 / Win
    rms = np.sqrt(np.maximum(S2 / Win - mean ** 2, 0))
    n_eff = np.where(W2 > 0, W ** 2 / np.where(W2 > 0, W2, 1), 0.0)
    out_frac = (Wlo + Whi) / W
    ptmean = SG / W
median = np.full(nb, np.nan)
for b in range(nb):
    if W[b] > 0:
        cum = Wlo[b] + np.r_[0.0, np.cumsum(sw_f[b])]
        if cum[0] < 0.5 * W[b] <= cum[-1]:
            median[b] = np.interp(0.5 * W[b], cum, R)

# --- rebin the fine histograms to the plotted binning
fine = nfine // a.bins
sw = sw_f.reshape(nb, a.bins, fine).sum(-1)
sw2 = sw2_f.reshape(nb, a.bins, fine).sum(-1)
cnt = cnt_f.reshape(nb, a.bins, fine).sum(-1)
edges = R[::fine]
centres = 0.5 * (edges[1:] + edges[:-1])

# --- summary table (stdout and summary.txt)
lines = ["%-15s %-13s %10s %11s %9s %7s %7s %7s %6s  %s"
         % ("pt_gen [GeV]", "|%s|" % var, "rows", "sum w [pb]", "n_eff", "mean", "rms", "median", "out%",
            "dominant H_T slice (share of sum w)")]
for ipt in range(npt):
    for ie in range(neta):
        b = ipt * neta + ie
        if n[b] == 0:
            continue
        j = int(np.argmax(slice_tab[:, 1, b]))
        lines.append("%6g-%-8g %5.3g-%-7.3g %10d %11.4g %9.0f %7.4f %7.4f %7.4f %6.2f  %s (%.2f)"
                     % (PT[ipt], PT[ipt + 1], ETA[ie], ETA[ie + 1], n[b], W[b], n_eff[b], mean[b], rms[b],
                        median[b], 100 * out_frac[b], slice_names[j], slice_tab[j, 1, b] / W[b]))
print("\n" + "\n".join(lines))
with open(os.path.join(outdir, "summary.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- one figure per cell
n_fig = 0
for ipt in range(npt):
    for ie in range(neta):
        b = ipt * neta + ie
        ok = cnt[b] > 0
        if n[b] < a.min_rows or not ok.any():
            continue
        norm = W[b] * np.diff(edges)
        dens, err = sw[b] / norm, np.sqrt(sw2[b]) / norm
        gauss = np.exp(-0.5 * ((centres - mean[b]) / rms[b]) ** 2) / (np.sqrt(2 * np.pi) * rms[b]) \
            if rms[b] > 0 else np.full_like(centres, np.nan)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, log in zip(axes, (False, True)):
            ax.errorbar(centres[ok], dens[ok], yerr=err[ok], fmt=".", ms=3, color="k", lw=0.8, label="jets (weighted)")
            ax.plot(centres, gauss, color="C0", lw=1, label="Gaussian, mean %.3f rms %.3f" % (mean[b], rms[b]))
            ax.set_xlabel(r"$p_T^{\rm rec}/p_T^{\rm gen}$")
            ax.set_ylabel("1/N dN/dr")
            if log:
                ax.set_yscale("log"); ax.set_ylim(max(dens[ok].min() / 3, 1e-7), dens[ok].max() * 3)
            ax.legend(fontsize=8)
        fig.suptitle(r"UL18 MadGraph, $%g \leq p_T^{\rm gen} < %g$ GeV, $%g \leq |%s| < %g$;  %d jets, $n_{\rm eff}$ = %.0f"
                     % (PT[ipt], PT[ipt + 1], ETA[ie], var, ETA[ie + 1], n[b], n_eff[b]), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "response_pt%02d_eta%02d__pt%g-%g_%s%g-%g.png"
                                 % (ipt, ie, PT[ipt], PT[ipt + 1], var, ETA[ie], ETA[ie + 1])), dpi=150)
        plt.close(fig)
        n_fig += 1

# --- overview: mean response and rms/mean vs <pt_gen>, one line per |eta| bin
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ie in range(neta):
    bs = np.arange(npt) * neta + ie
    m = n[bs] >= max(a.min_rows, 1)
    if not m.any():
        continue
    lab = r"$%g \leq |%s| < %g$" % (ETA[ie], var, ETA[ie + 1])
    axes[0].plot(ptmean[bs][m], mean[bs][m], "o-", ms=3, lw=1, label=lab)
    axes[1].plot(ptmean[bs][m], (rms / mean)[bs][m], "o-", ms=3, lw=1, label=lab)
for ax, yl in zip(axes, ("weighted mean response", "rms / mean")):
    ax.set_xscale("log"); ax.set_xlabel(r"$\langle p_T^{\rm gen} \rangle$ [GeV]"); ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
axes[1].legend(fontsize=7)
fig.suptitle("UL18 MadGraph, inside %g <= r < %g" % (edges[0], edges[-1]), fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "summary.png"), dpi=150)
plt.close(fig)
print("\n%d cell figures, summary.png and summary.txt in %s" % (n_fig, outdir))
