#!/usr/bin/env python3
"""THE ONLY DARWIN-DEPENDENT FILE.  One M4 input file -> one npz of per-jet rows.

Runs inside the EL9 container with the Darwin/Core dictionaries on the
library path (see run_export.sh); everything downstream reads the npz with
numpy alone.  Schema: jerfit/rows.py.

    python3 export_response.py <M4 file>.root --out <dir> [--max-events N]
        [--genpt-min 20] [--max-dr 0.8] [--prescale K]

What is read, and why (from the campaign handover, docs/jer_unbinned_handover.md
of schoef/0070):

  * stage M4_applyJEScorrections: fully JES-corrected, BEFORE applyJERsmearing
    (M5 would carry the very scale factor a JER measurement determines);
  * genEvent.Weight()  -- a cross section in pb (applyMClumi has multiplied by
    sigma/sum w), so the H_T slices add without any further factor; at M4 it
    does NOT contain the pile-up weight;
  * jet.Weight() == 0  -- the jet is REJECTED (jet ID, veto map); never written;
  * jet.CorrPt(), CorrP4() -- JES-corrected (p4 * scales[0]);
    jet.p4.Rapidity()      -- Darwin's Rapidity() uses the RAW p4; kept so;
  * one file at a time: a TChain over M4 files segfaults (sliced friend
    elements), and Flow from PyROOT dies with bad_alloc.

The matching radius is NOT applied here beyond a loose cap: every accepted
reconstructed jet is paired with its nearest generator jet and the distance
is written, so the analysis cuts at R/2 (or 0.2) and varies it.
"""
import argparse, os, sys, time, math
from array import array
import numpy as np

import ROOT
ROOT.gSystem.Load("libDarwin")
ROOT.gSystem.Load("libCoreObjects")
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError
from ROOT.Math import VectorUtil


def parse_meta(path):
    """.../M4_applyJEScorrections/QCD_HT300to500_MadGraph_UL18/3.root"""
    sample = os.path.basename(os.path.dirname(os.path.abspath(path)))
    parts = sample.split("_")
    meta = dict(source=os.path.abspath(path), sample=sample, era="", slice="", generator="")
    if len(parts) >= 4 and parts[0] == "QCD" and parts[1].startswith("HT"):
        meta.update(slice=parts[1][2:], generator=parts[2], era=parts[3])
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--max-events", type=int, default=-1)
    ap.add_argument("--genpt-min", type=float, default=20.0,
                    help="generator jets below this are not considered [GeV]")
    ap.add_argument("--max-dr", type=float, default=0.8,
                    help="pairs beyond this are not written (loose cap; cut later)")
    ap.add_argument("--prescale", type=int, default=1,
                    help="keep every K-th event and multiply its weight by K")
    ap.add_argument("--era", help="override the era parsed from the path")
    a = ap.parse_args()

    meta = parse_meta(a.input)
    if a.era:
        meta["era"] = a.era
    f = ROOT.TFile.Open(a.input)
    if not f or f.IsZombie():
        sys.exit("unreadable: " + a.input)
    t = f.Get("events")
    n_all = t.GetEntries()
    n = n_all if a.max_events < 0 else min(a.max_events, n_all)

    # typed arrays, not Python lists: a file holds up to ~6M events and ~20M
    # pairs, and a list of Python floats costs ~300 bytes per pair (4 GB OOM)
    cols = {k: array("d") for k in ("response", "genpt", "geny", "recpt", "recy", "weight",
                                    "w_event", "dr", "dr2")}
    gen = {k: array("d") for k in ("g_pt", "g_y", "g_dr")}
    sumw = 0.0
    n_read = n_jets_acc = n_jets_rej = 0
    t0 = time.time()
    for i in range(0, n, a.prescale):
        t.GetEntry(i)
        wevt = t.genEvent.Weight() * a.prescale
        sumw += wevt
        n_read += 1

        gens = [(gj.p4.Pt(), gj.p4.Rapidity(), gj.p4) for gj in t.genJets
                if gj.p4.Pt() >= a.genpt_min]
        recs = []
        for jet in t.recJets:
            wjet = jet.Weight()
            if wjet == 0.:                      # rejected, not down-weighted
                n_jets_rej += 1
                continue
            n_jets_acc += 1
            recs.append((jet.CorrPt(), jet.p4.Rapidity(), jet.CorrP4(), wjet))

        # all pair distances once; row minima -> rec matching, column minima
        # -> gen efficiency
        drs = [[VectorUtil.DeltaR(g[2], r[2]) for g in gens] for r in recs]
        for ir, r in enumerate(recs):
            if not gens:
                break
            d = drs[ir]
            j = min(range(len(d)), key=d.__getitem__)
            if d[j] >= a.max_dr:
                continue
            d2 = min((x for k, x in enumerate(d) if k != j), default=math.inf)
            gpt, gy = gens[j][0], gens[j][1]
            cols["response"].append(r[0] / gpt)
            cols["genpt"].append(gpt); cols["geny"].append(gy)
            cols["recpt"].append(r[0]); cols["recy"].append(r[1])
            cols["weight"].append(wevt * r[3]); cols["w_event"].append(wevt)
            cols["dr"].append(d[j]); cols["dr2"].append(d2)
        for jg, g in enumerate(gens):
            gen["g_pt"].append(g[0]); gen["g_y"].append(g[1])
            gen["g_dr"].append(min((drs[ir][jg] for ir in range(len(recs))), default=math.inf))

        if n_read % 20000 == 0:
            print("  %d / %d events, %.0f ev/s" % (n_read, n // a.prescale, n_read / (time.time() - t0)),
                  flush=True)
    f.Close()

    out = {k: np.frombuffer(v, np.float64).astype(np.float32) for k, v in cols.items()}
    out["weight"] = np.frombuffer(cols["weight"], np.float64)   # pb, keep precision
    out["w_event"] = np.frombuffer(cols["w_event"], np.float64)
    out.update({k: np.frombuffer(v, np.float64).astype(np.float32) for k, v in gen.items()})
    out.update(era=meta["era"], slice=meta["slice"], generator=meta["generator"],
               sample=meta["sample"], source=meta["source"],
               n_events=n_read, n_events_file=n_all, sumw_events=sumw,
               n_jets_accepted=n_jets_acc, n_jets_rejected=n_jets_rej,
               genpt_min=a.genpt_min, max_dr=a.max_dr, prescale=a.prescale)
    os.makedirs(a.out, exist_ok=True)
    stem = "%s_%s" % (meta["sample"], os.path.splitext(os.path.basename(a.input))[0])
    path = os.path.join(a.out, stem + ".npz")
    np.savez_compressed(path, **out)
    print("events %d  sum w %.6g pb  jets accepted %d rejected %d  pairs %d  gen jets %d"
          % (n_read, sumw, n_jets_acc, n_jets_rej, len(out["response"]), len(out["g_pt"])))
    print("wrote", path, "in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
