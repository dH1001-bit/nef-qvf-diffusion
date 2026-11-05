#!/usr/bin/env python3
"""The weighted jet response r = pt_rec / pt_gen in EVERY cell of a pt_gen x |eta|
grid, read straight from the campaign export (all 72 files, every jet) in a
single pass, and plotted cell by cell (linear and log scale) into one directory.
numpy + matplotlib only; written top to bottom so it can be stepped through in
ipython.  plot_response.py does the same for one hand-picked window.

    python3 scripts/plot_response_all_bins.py
    python3 scripts/plot_response_all_bins.py --min-neff 0                # no merging: the plain grid
    python3 scripts/plot_response_all_bins.py --var geneta --outdir fig/resp_geneta
    python3 scripts/plot_response_all_bins.py --eta-edges 0 0.522 0.783 1.131 1.305 1.740 1.930 2.043 2.322 2.5 2.853 2.964 3.139 5.191
    python3 scripts/plot_response_all_bins.py --load fig/response_bins_receta/histograms.npz --min-neff 300   # re-merge, no reading
    ipython -i scripts/plot_response_all_bins.py -- --max-files 9

Default grid: the pt_gen and |y| edges of count_jets.py.  The --eta-edges
example is the CMS JER scale-factor |eta| binning.

Merging (--min-neff): n_eff falls steeply towards high |eta|, so within each
pt_gen row adjacent |eta| cells are merged, walking outwards from |eta| = 0:
a cell is closed once its n_eff reaches --min-neff, and a leftover tail that
never reaches it is added to the last closed cell.  The pt_gen axis is never
merged, so every row keeps its pt_gen window and the rows can end up with
different |eta| edges.  These are written to binning.json.  A row whose total
n_eff is below the threshold becomes one cell and is marked LOW in summary.txt.

How it works: the files are read once.  Each jet gets a cell index
(ipt, ieta) and a fine response-bin index, and np.bincount adds w, w^2 and 1
into per-cell histograms.  Memory is one file plus the histograms, never the
jets themselves.  The fine, UNMERGED histograms (--fine sub-bins per plotted
bin) are saved to <outdir>/histograms.npz.  Every stored quantity is a sum,
so merging happens afterwards, and --load can re-merge (another --min-neff)
or rebin (another --bins that divides the stored bin count) without the
~10 min read.

Mean and rms are exact (from sum w r and sum w r^2 over the jets inside
--range); the median is interpolated in the fine histogram.

As in plot_response.py: the grid is in GENERATOR pt (binning in reconstructed
pt selects on the fluctuation being measured), no trigger is needed (the H_T
slices with their cross-section weights add up to the inclusive spectrum),
and every slice must be read.  summary.txt lists, per cell, which slice
dominates the summed weight.
"""
import argparse, glob, json, os, re, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pt-edges", nargs="+", type=float,
                default=[97, 140, 200, 280, 400, 550, 750, 1000, 1400, 2000, 3500, 1e5],
                help="pt_gen bin edges [GeV]")
ap.add_argument("--eta-edges", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
                help="finest bin edges in |var|; merged per pt_gen row according to --min-neff")
ap.add_argument("--min-neff", type=float, default=100.0,
                help="merge adjacent |var| cells of a pt_gen row until n_eff reaches this (0: no merging)")
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

# --- merge adjacent |eta| cells within each pt_gen row until n_eff >= --min-neff
def neff_of(w, w2):
    return w * w / w2 if w2 > 0 else 0.0

cells = []                                        # (ipt, ie0, ie1): |eta| from ETA[ie0] to ETA[ie1 + 1]
for ipt in range(npt):
    row, start, gw, gw2 = [], 0, 0.0, 0.0
    for ie in range(neta):
        b = ipt * neta + ie
        gw += W[b]; gw2 += W2[b]
        if neff_of(gw, gw2) >= a.min_neff:
            row.append([start, ie]); start, gw, gw2 = ie + 1, 0.0, 0.0
    if start < neta:                              # tail that never reached the threshold
        if row:
            row[-1][1] = neta - 1
        else:
            row.append([start, neta - 1])
    cells += [(ipt, s, e) for s, e in row]
nc = len(cells)

def merge(x):
    """(nb, ...) per fine cell -> (nc, ...) per merged cell: every stored quantity is a sum."""
    x = x.reshape(npt, neta, *x.shape[1:])
    return np.array([x[ipt, s:e + 1].sum(0) for ipt, s, e in cells])

sw_c, sw2_c, cnt_c, n_c, W_c, W2_c, Wlo_c, Whi_c, S1_c, S2_c, SG_c = map(
    merge, (sw_f, sw2_f, cnt_f, n, W, W2, Wlo, Whi, S1, S2, SG))
slice_c = merge(np.moveaxis(slice_tab, -1, 0))   # (nc, n_slices, 3)
print("\n%d fine cells -> %d cells after merging |%s| to n_eff >= %g" % (nb, nc, var, a.min_neff))

eta_edges_per_pt = [[float(ETA[0])] for _ in range(npt)]   # the merged |eta| edges of each pt_gen row
for ipt, s, e in cells:
    eta_edges_per_pt[ipt].append(float(ETA[e + 1]))
binning = dict(var=var, min_neff=a.min_neff, pt_edges=[float(x) for x in PT], eta_edges_per_pt=eta_edges_per_pt)
with open(os.path.join(outdir, "binning.json"), "w") as fh:
    json.dump(binning, fh, indent=1)

# --- per-cell summary numbers
with np.errstate(invalid="ignore", divide="ignore"):
    Win = W_c - Wlo_c - Whi_c
    mean = S1_c / Win
    rms = np.sqrt(np.maximum(S2_c / Win - mean ** 2, 0))
    n_eff = np.where(W2_c > 0, W_c ** 2 / np.where(W2_c > 0, W2_c, 1), 0.0)
    out_frac = (Wlo_c + Whi_c) / W_c
    ptmean = SG_c / W_c
median = np.full(nc, np.nan)
for c in range(nc):
    if W_c[c] > 0:
        cum = Wlo_c[c] + np.r_[0.0, np.cumsum(sw_c[c])]
        if cum[0] < 0.5 * W_c[c] <= cum[-1]:
            median[c] = np.interp(0.5 * W_c[c], cum, R)

# --- rebin the fine response histograms to the plotted binning
fine = nfine // a.bins
sw = sw_c.reshape(nc, a.bins, fine).sum(-1)
sw2 = sw2_c.reshape(nc, a.bins, fine).sum(-1)
cnt = cnt_c.reshape(nc, a.bins, fine).sum(-1)
edges = R[::fine]
centres = 0.5 * (edges[1:] + edges[:-1])

# --- summary table (stdout and summary.txt)
lines = ["%-15s %-13s %10s %11s %9s %7s %7s %7s %6s  %-6s %s"
         % ("pt_gen [GeV]", "|%s|" % var, "rows", "sum w [pb]", "n_eff", "mean", "rms", "median", "out%",
            "merged", "dominant H_T slice (share of sum w)")]
for c, (ipt, s, e) in enumerate(cells):
    if n_c[c] == 0:
        continue
    j = int(np.argmax(slice_c[c, :, 1]))
    flag = ("%d" % (e - s + 1) if e > s else "") + (" LOW" if n_eff[c] < a.min_neff else "")
    lines.append("%6g-%-8g %5.3g-%-7.3g %10d %11.4g %9.0f %7.4f %7.4f %7.4f %6.2f  %-6s %s (%.2f)"
                 % (PT[ipt], PT[ipt + 1], ETA[s], ETA[e + 1], n_c[c], W_c[c], n_eff[c], mean[c], rms[c],
                    median[c], 100 * out_frac[c], flag, slice_names[j], slice_c[c, j, 1] / W_c[c]))
print("\n" + "\n".join(lines))
with open(os.path.join(outdir, "summary.txt"), "w") as fh:
    fh.write("# merged: number of fine |%s| cells combined; LOW: n_eff still below --min-neff %g\n"
             % (var, a.min_neff))
    fh.write("\n".join(lines) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- one figure per (merged) cell
n_fig = 0
for c, (ipt, s, e) in enumerate(cells):
    ok = cnt[c] > 0
    if n_c[c] < a.min_rows or not ok.any():
        continue
    norm = W_c[c] * np.diff(edges)
    dens, err = sw[c] / norm, np.sqrt(sw2[c]) / norm
    gauss = np.exp(-0.5 * ((centres - mean[c]) / rms[c]) ** 2) / (np.sqrt(2 * np.pi) * rms[c]) \
        if rms[c] > 0 else np.full_like(centres, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, log in zip(axes, (False, True)):
        ax.errorbar(centres[ok], dens[ok], yerr=err[ok], fmt=".", ms=3, color="k", lw=0.8, label="jets (weighted)")
        ax.plot(centres, gauss, color="C0", lw=1, label="Gaussian, mean %.3f rms %.3f" % (mean[c], rms[c]))
        ax.set_xlabel(r"$p_T^{\rm rec}/p_T^{\rm gen}$")
        ax.set_ylabel("1/N dN/dr")
        if log:
            ax.set_yscale("log"); ax.set_ylim(max(dens[ok].min() / 3, 1e-7), dens[ok].max() * 3)
        ax.legend(fontsize=8)
    note = "  (%d |%s| cells merged)" % (e - s + 1, var) if e > s else ""
    fig.suptitle(r"UL18 MadGraph, $%g \leq p_T^{\rm gen} < %g$ GeV, $%g \leq |%s| < %g$;  %d jets, $n_{\rm eff}$ = %.0f%s"
                 % (PT[ipt], PT[ipt + 1], ETA[s], var, ETA[e + 1], n_c[c], n_eff[c], note), fontsize=10)
    fig.tight_layout()
    tag = "eta%02d" % s if s == e else "eta%02d-%02d" % (s, e)
    fig.savefig(os.path.join(outdir, "response_pt%02d_%s__pt%g-%g_%s%g-%g.png"
                             % (ipt, tag, PT[ipt], PT[ipt + 1], var, ETA[s], ETA[e + 1])), dpi=150)
    plt.close(fig)
    n_fig += 1

# --- overview: mean response and rms/mean vs <pt_gen>; one line per distinct |eta| range,
#     coloured by its |eta| centre (merged ranges only connect the pt rows that share them)
series = {}
for c, (ipt, s, e) in enumerate(cells):
    if n_c[c] >= max(a.min_rows, 1):
        series.setdefault((s, e), []).append(c)
cmap = plt.get_cmap("viridis")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for (s, e), cs in sorted(series.items()):
    col = cmap(0.5 * (ETA[s] + ETA[e + 1]) / ETA[-1])
    lab = r"$%g \leq |%s| < %g$" % (ETA[s], var, ETA[e + 1])
    style = "o-" if s == e else "s--"
    axes[0].plot(ptmean[cs], mean[cs], style, ms=3, lw=1, color=col, label=lab)
    axes[1].plot(ptmean[cs], (rms / mean)[cs], style, ms=3, lw=1, color=col, label=lab)
for ax, yl in zip(axes, ("weighted mean response", "rms / mean")):
    ax.set_xscale("log"); ax.set_xlabel(r"$\langle p_T^{\rm gen} \rangle$ [GeV]"); ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
axes[1].legend(fontsize=6, ncol=2)
fig.suptitle("UL18 MadGraph, inside %g <= r < %g;  |%s| merged to n_eff >= %g (squares: merged cells)"
             % (edges[0], edges[-1], var, a.min_neff), fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "summary.png"), dpi=150)
plt.close(fig)
print("\n%d cell figures, summary.png, summary.txt and binning.json in %s" % (n_fig, outdir))
