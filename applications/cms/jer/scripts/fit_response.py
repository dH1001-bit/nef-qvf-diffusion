#!/usr/bin/env python3
"""Fit the jet response from exported rows: one bin like the starter, the
per-bin table, the binned NSC reference, and the global unbinned NSC fits.

    python3 scripts/fit_response.py 'rows/UL18/*.npz' --absy 0 0.5 --out fig/UL18
    python3 scripts/fit_response.py rows/toy/*.npz --absy 0 2.5 --out fig/toy

Weights are flattened in pt for the global fits (see jerfit.rows.flatten).
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from jerfit import concatenate, select, flatten, fit_bin, fit_core, fit_nsc, GaussianResponse, matching_efficiency
from jerfit.nsc import fit_per_bin, fit_nsc_to_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="npz files or globs")
    ap.add_argument("--absy", nargs=2, type=float, default=(0.0, 0.5), metavar=("LO", "HI"))
    ap.add_argument("--genpt", nargs=2, type=float, default=(30.0, 3500.0), metavar=("LO", "HI"))
    ap.add_argument("--max-dr", type=float, default=0.4, help="matching radius (JME: R/2)")
    ap.add_argument("--isolate", type=float, default=None, help="second-nearest gen jet must be beyond this")
    ap.add_argument("--response", nargs=2, type=float, default=(0.0, 2.0))
    ap.add_argument("--core", type=float, default=1.5, help="core window in sigma")
    ap.add_argument("--bin", nargs=2, type=float, default=(300.0, 500.0), help="the single-bin figure")
    ap.add_argument("--edges", type=float, nargs="+",
                    default=[30, 40, 55, 75, 100, 140, 200, 280, 400, 550, 750, 1000, 1400, 2000, 3500])
    ap.add_argument("--out", default=None, help="directory for figures and results.json")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    files = sorted(sum((glob.glob(p) or [p] for p in a.files), []))
    rows = concatenate(files)
    print("read %d file(s): %d pairs, %d gen jets, %d events, sum w(events) %.6g"
          % (len(files), len(rows), 0 if rows.gen is None else len(rows.gen["g_pt"]),
             rows.meta["n_events"], rows.meta["sumw_events"]))
    sel = select(rows, genpt=tuple(a.genpt), absy=tuple(a.absy), max_dr=a.max_dr,
                 isolate_dr2=a.isolate, response=tuple(a.response))
    print("selected %d pairs, n_eff %.0f (|y| in [%g,%g), genpt in [%g,%g), dR < %g)"
          % (len(sel), sel.n_eff, *a.absy, *a.genpt, a.max_dr))
    results = dict(selection=vars(a), n=len(sel), n_eff=sel.n_eff)

    # 1. the single bin, as the starter does it
    b = select(sel, genpt=tuple(a.bin))
    if len(b) < 50:
        sys.exit("only %d jets in the bin %g <= genpt < %g; choose --bin inside the populated range" % (len(b), *a.bin))
    full = fit_bin(b.response, b.weight); core = fit_core(b.response, b.weight, k=a.core)
    print("\n[bin %g <= genpt < %g, %d jets, n_eff %.0f]" % (*a.bin, len(b), b.n_eff))
    print("Gaussian, full range  : mu = %.4f +- %.4f   sigma = %.4f +- %.4f"
          % (full["mu"], full.error("mu"), full["sigma"], full.error("sigma")))
    print("Gaussian, +-%gs core : mu = %.4f +- %.4f   sigma = %.4f +- %.4f  (%d iterations)"
          % (a.core, core["mu"], core.error("mu"), core["sigma"], core.error("sigma"), core.extra["iterations"]))
    results["bin"] = dict(lo=a.bin[0], hi=a.bin[1], n=len(b),
                          full=dict(zip(full.names, full.values.tolist())), full_err=full.errors.tolist(),
                          core=dict(zip(core.names, core.values.tolist())), core_err=core.errors.tolist())

    # 2. per-bin table and binned NSC reference
    pts_full = fit_per_bin(sel, a.edges); pts_core = fit_per_bin(sel, a.edges, core_k=a.core)
    print("\n[per bin]   pt      n    n_eff   sigma(full)          sigma(core)")
    for pf, pc in zip(pts_full, pts_core):
        print("  %7.1f %7d %7.0f   %.4f +- %.4f   %.4f +- %.4f"
              % (pf["pt"], pf["n"], pf["fit"].n_eff, pf["fit"]["sigma"], pf["fit"].error("sigma"),
                 pc["fit"]["sigma"], pc["fit"].error("sigma")))
    ref_full = fit_nsc_to_points(pts_full); ref_core = fit_nsc_to_points(pts_core)
    print("\n[binned NSC, full-range widths, chi2/ndf %.1f/%d]\n%s" % (ref_full.extra["chi2"], ref_full.extra["ndf"], ref_full))
    print("\n[binned NSC, core widths, chi2/ndf %.1f/%d]\n%s" % (ref_core.extra["chi2"], ref_core.extra["ndf"], ref_core))

    # 3. global unbinned fits on flattened weights
    flat = flatten(sel)
    g_full = fit_nsc(flat, GaussianResponse())
    g_core = fit_nsc(flat, GaussianResponse(k=a.core))
    print("\n[unbinned NSC, full range]\n%s" % g_full)
    print("\n[unbinned NSC, +-%g sigma(pt) window, %d iterations]\n%s" % (a.core, g_core.extra["iterations"], g_core))
    for key, res in (("binned_full", ref_full), ("binned_core", ref_core),
                     ("unbinned_full", g_full), ("unbinned_core", g_core)):
        results[key] = dict(values=dict(zip(res.names, res.values.tolist())), errors=res.errors.tolist(),
                            converged=res.converged, n=res.n, n_eff=res.n_eff)

    # 4. matching efficiency from the gen table
    if rows.gen is not None:
        c, eff, n = matching_efficiency(rows, a.edges, max_dr=a.max_dr, absy_max=a.absy[1])
        print("\n[matching efficiency, dR < %g, |y| < %g]" % (a.max_dr, a.absy[1]))
        for ci, ei, ni in zip(c, eff, n):
            if ni: print("  %7.1f  %.3f  (%d gen jets)" % (ci, ei, ni))
        results["efficiency"] = dict(pt=c.tolist(), eff=eff.tolist(), n=n.tolist())

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "results.json"), "w") as f:
            json.dump(results, f, indent=1, default=float)
        if not a.no_plots:
            from jerfit import plots
            tag = "%s  %g <= |y| < %g" % (",".join(rows.meta["eras"]), *a.absy)
            plots.response_bin(b, full, core, title="%s, %g <= genpt < %g GeV, %d jets" % (tag, *a.bin, len(b)),
                               path=os.path.join(a.out, "response_bin.pdf"))
            plots.sigma_vs_pt(pts_full, {"binned NSC (full)": ref_full.values, "unbinned NSC (full)": g_full.values[1:],
                                         "unbinned NSC (core)": g_core.values[1:]},
                              title=tag + ", full-range widths", path=os.path.join(a.out, "sigma_vs_pt_full.pdf"))
            plots.sigma_vs_pt(pts_core, {"binned NSC (core)": ref_core.values, "unbinned NSC (core)": g_core.values[1:]},
                              title=tag + ", core widths", path=os.path.join(a.out, "sigma_vs_pt_core.pdf"))
            plots.pull_vs_pt(flat, g_full.values, a.edges, title=tag + ", full-range NSC", path=os.path.join(a.out, "pull_full.pdf"))
            plots.pull_vs_pt(flat, g_core.values, a.edges, k=a.core, title=tag + ", core NSC (inside window)",
                             path=os.path.join(a.out, "pull_core.pdf"))
            if rows.gen is not None:
                plots.efficiency(c, eff, n, title=tag, path=os.path.join(a.out, "efficiency.pdf"))
            print("\nfigures and results.json in", a.out)


if __name__ == "__main__":
    main()
