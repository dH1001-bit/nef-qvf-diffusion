#!/usr/bin/env python3
"""Reduce exported rows file by file into one npz: apply the analysis cuts
(rapidity, generator pt, matching radius) and optionally a prescale, so that
a fit-sized sample leaves the cluster.  numpy only.

    python3 scripts/reduce_rows.py 'rows/UL18/*.npz' out/UL18_absy0-0.5.npz --absy 0 0.5
    python3 scripts/reduce_rows.py 'rows/UL18/*.npz' out/UL18_student.npz --prescale 50

The prescale keeps every K-th row (deterministic, by row index within each
file) and multiplies the weight by K, so sums of weights are preserved in
expectation.  The generator-jet table is reduced with the same cuts on |y|
and the same prescale.  Per-file metadata (event counts, summed event
weights) is summed.
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from jerfit.rows import load, PAIR_COLUMNS

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+"); ap.add_argument("out")
ap.add_argument("--absy", nargs=2, type=float, default=(0.0, 99.0))
ap.add_argument("--genpt", nargs=2, type=float, default=(0.0, 1e9))
ap.add_argument("--max-dr", type=float, default=None, help="default: keep the export's radius")
ap.add_argument("--prescale", type=int, default=1)
a = ap.parse_args()

files = sorted(sum((glob.glob(p) or [p] for p in a.files), []))
cols, gen = {}, {}
n_events = 0; sumw = 0.0; eras, slices = set(), set()
n_in = n_out = 0
for i, f in enumerate(files):
    r = load(f)
    ay = np.abs(r.recy)
    m = (ay >= a.absy[0]) & (ay < a.absy[1]) & (r.genpt >= a.genpt[0]) & (r.genpt < a.genpt[1])
    if a.max_dr is not None:
        m &= r.dr < a.max_dr
    idx = np.flatnonzero(m)[::a.prescale]
    n_in += len(r); n_out += len(idx)
    for k in r.columns:
        cols.setdefault(k, []).append(getattr(r, k)[idx])
    cols.setdefault("file_id", []).append(np.full(len(idx), i, np.int16))
    if r.gen is not None:
        gy = np.abs(r.gen["g_y"]); gm = (gy >= a.absy[0]) & (gy < a.absy[1])
        gidx = np.flatnonzero(gm)[::a.prescale]
        for k, v in r.gen.items():
            gen.setdefault(k, []).append(v[gidx])
    n_events += int(r.meta.get("n_events", 0)); sumw += float(r.meta.get("sumw_events", 0.0))
    eras.add(str(r.meta.get("era", ""))); slices.add(str(r.meta.get("slice", "")))
    print("  %-45s %9d -> %8d" % (os.path.basename(f), len(r), len(idx)), flush=True)

out = {k: np.concatenate(v) for k, v in cols.items()}
out["weight"] = out["weight"] * a.prescale
out.update({k: np.concatenate(v) for k, v in gen.items()})
out.update(era=",".join(sorted(eras)), slice=",".join(sorted(slices)), source="reduced from %d files" % len(files),
           n_events=n_events, sumw_events=sumw, prescale=a.prescale, absy_lo=a.absy[0], absy_hi=a.absy[1],
           genpt_lo=a.genpt[0], genpt_hi=a.genpt[1], files=np.array([os.path.basename(f) for f in files]))
np.savez_compressed(a.out, **out)
print("rows %d -> %d, gen jets %d, events %d, sum w(events) %.6g pb -> %s (%.0f MB)"
      % (n_in, n_out, len(out.get("g_pt", [])), n_events, sumw, a.out, os.path.getsize(a.out) / 1e6))
