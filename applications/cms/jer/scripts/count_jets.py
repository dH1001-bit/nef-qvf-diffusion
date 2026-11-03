#!/usr/bin/env python3
"""Run over EVERY jet of the campaign export, one file at a time, and tally
rows, summed weight and effective sample size by H_T slice, |y| and pt_gen.
numpy only; memory is one file at a time (a few GB at most), never the
whole sample.

    python3 scripts/count_jets.py                       # all 72 files, ~10 min on cbe
    python3 scripts/count_jets.py --max-files 3         # a quick look at the first three files

Default input: /scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz

What this is for: the export has ~5e8 rows, but rows are not statistics.
Compare the 'rows' and 'n_eff' columns of every table.
"""
import argparse, glob, os, re, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="*",
                default=["/scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz"])
ap.add_argument("--max-files", type=int, default=None)
a = ap.parse_args()

files = sorted(sum((glob.glob(p) for p in a.files), []))
if a.max_files:
    files = files[:a.max_files]
if not files:
    raise SystemExit("no input files: %s" % a.files)

Y_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0])
PT_EDGES = np.array([97, 140, 200, 280, 400, 550, 750, 1000, 1400, 2000, 3500, 1e5])
R_EDGES = np.linspace(0.0, 2.0, 201)

def tally(n):
    return dict(n=np.zeros(n, np.int64), w=np.zeros(n), w2=np.zeros(n))

def add(t, idx, w, n):
    t["n"] += np.bincount(idx, minlength=n)[:n]
    t["w"] += np.bincount(idx, weights=w, minlength=n)[:n]
    t["w2"] += np.bincount(idx, weights=w * w, minlength=n)[:n]

by_slice = {}
by_y = tally(len(Y_EDGES) - 1)
by_pt = tally(len(PT_EDGES) - 1)
resp_hist = np.zeros(len(R_EDGES) - 1)
n_events = n_veto = 0
sumw_events = 0.0
n_resp_out = 0
t0 = time.time()

for i, f in enumerate(files):
    with np.load(f) as z:
        genpt = z["genpt"]; recpt = z["recpt"]; recy = z["recy"]; w = z["weight"].astype(np.float64)
        n_events += int(z["n_events_read"]); n_veto += int(z["n_events_vetoed"])
        sumw_events += float(z["sum_weights"])
        sl = str(z["slice"])
    n = len(w)
    t = by_slice.setdefault(sl, dict(n=0, w=0.0, w2=0.0))
    t["n"] += n; t["w"] += w.sum(); t["w2"] += (w * w).sum()
    iy = np.clip(np.digitize(np.abs(recy), Y_EDGES) - 1, 0, len(Y_EDGES) - 2)
    add(by_y, iy, w, len(Y_EDGES) - 1)
    ip = np.clip(np.digitize(genpt, PT_EDGES) - 1, 0, len(PT_EDGES) - 2)
    add(by_pt, ip, w, len(PT_EDGES) - 1)
    r = recpt.astype(np.float64) / genpt
    n_resp_out += int(np.sum(r >= R_EDGES[-1]))
    resp_hist += np.histogram(r, R_EDGES, weights=w)[0]
    print("  %3d/%d %-32s %10d rows  %5.0f s" % (i + 1, len(files), os.path.basename(f), n, time.time() - t0), flush=True)

def neff(w, w2):
    return np.where(w2 > 0, w * w / np.where(w2 > 0, w2, 1), 0)

def order(sl):
    m = re.match(r"(\d+)", sl); return int(m.group(1)) if m else 0

N = sum(t["n"] for t in by_slice.values())
W = sum(t["w"] for t in by_slice.values()); W2 = sum(t["w2"] for t in by_slice.values())
print("\n%d files, %d events read (%d vetoed), %d jet rows, sum w %.4g pb, n_eff %.4g  (%.1f min)"
      % (len(files), n_events, n_veto, N, W, float(neff(W, W2)), (time.time() - t0) / 60))
print("rows per event %.2f; the export keeps only matched jets with pt_gen > 97 GeV" % (N / max(n_events, 1)))

print("\nby H_T slice      rows        sum w [pb]    n_eff      w per row [pb]")
for sl in sorted(by_slice, key=order):
    t = by_slice[sl]
    print("  %-12s %11d  %12.4g  %10.4g   %.3g" % (sl, t["n"], t["w"], float(neff(t["w"], t["w2"])), t["w"] / max(t["n"], 1)))

print("\nby |y_rec|         rows        sum w [pb]    n_eff")
for k in range(len(Y_EDGES) - 1):
    print("  %3.1f-%-3.1f     %11d  %12.4g  %10.4g" % (Y_EDGES[k], Y_EDGES[k + 1], by_y["n"][k], by_y["w"][k],
                                                   float(neff(by_y["w"][k], by_y["w2"][k]))))

print("\nby pt_gen [GeV]    rows        sum w [pb]    n_eff")
for k in range(len(PT_EDGES) - 1):
    print("  %5.0f-%-6.0f %11d  %12.4g  %10.4g" % (PT_EDGES[k], PT_EDGES[k + 1], by_pt["n"][k], by_pt["w"][k],
                                                  float(neff(by_pt["w"][k], by_pt["w2"][k]))))

c = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
m = (resp_hist * c).sum() / resp_hist.sum()
print("\nresponse, all jets: weighted mean %.4f, rms %.4f; %d rows with response >= 2 (mis-associations, see README)"
      % (m, np.sqrt((resp_hist * (c - m) ** 2).sum() / resp_hist.sum()), n_resp_out))
