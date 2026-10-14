"""Closure of the Gaussian fits on the synthetic sample.  Run: python3 tests/test_closure.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from jerfit import synthetic, concatenate, select, flatten, fit_bin, fit_core, fit_nsc, GaussianResponse
from jerfit.nsc import fit_per_bin, fit_nsc_to_points
from jerfit.width import nsc_sigma

rows = concatenate(synthetic.make(n_per_slice=30000, seed=1))
rows = select(rows, max_dr=0.4)
T = synthetic.TRUE
print("rows %d  sum w %.4g  n_eff %.0f" % (len(rows), rows.sumw, rows.n_eff))

# 1. one bin, full-range Gaussian and core fit agree with the truth
b = select(rows, genpt=(300, 500))
s_true = float(np.average(nsc_sigma(b.genpt, T["N"], T["S"], T["C"]), weights=b.weight))
full = fit_bin(b.response, b.weight)
core = fit_core(b.response, b.weight, k=1.5)
print("\n[bin 300-500] truth sigma(avg) = %.5f" % s_true)
print("full:\n" + str(full)); print("core:\n" + str(core))
for name, f in (("full", full), ("core", core)):
    pull = (f["sigma"] - s_true) / f.error("sigma")
    assert abs(pull) < 4, "%s sigma pull %.1f" % (name, pull)
    assert abs((f["mu"] - 1) / f.error("mu")) < 4

# 2. global unbinned NSC fit recovers N, S, C.  Shown twice: with the raw
#    cross-section weights (consistent but inefficient: the likelihood is the
#    lowest-pt rows) and with the weights flattened in pt.
g_raw = fit_nsc(rows, GaussianResponse())
print("\n[global NSC, full range, raw cross-section weights]\n" + str(g_raw))
flat = flatten(rows)
g = fit_nsc(flat, GaussianResponse())
print("\n[global NSC, full range, flattened weights]\n" + str(g))
for k in ("mu", "N", "S", "C"):
    pull = (g[k] - T[k]) / g.error(k)
    print("   %s pull %+.2f" % (k, pull))
    assert abs(pull) < 4, "%s pull %.1f" % (k, pull)

# 3. windowed global fit, same answer on a Gaussian truth
gk = fit_nsc(flat, GaussianResponse(k=2.0))
print("\n[global NSC, +-2 sigma window, %d iterations]\n" % gk.extra["iterations"] + str(gk))
for k in ("N", "S", "C"):
    assert abs((gk[k] - T[k]) / gk.error(k)) < 4

# 4. binned reference
edges = [30, 50, 80, 120, 180, 260, 380, 550, 800, 1200, 1800, 3500]
pts = fit_per_bin(rows, edges)
ref = fit_nsc_to_points(pts)
print("\n[binned NSC reference, chi2/ndf %.2f/%d]\n" % (ref.extra["chi2"], ref.extra["ndf"]) + str(ref))

# 5. the Gaussian fit is biased once a tail is present -- the motivation
rows_t = flatten(select(concatenate(synthetic.make(n_per_slice=30000, seed=2, tail_fraction=0.03))))
gt = fit_nsc(rows_t, GaussianResponse())
gtk = fit_nsc(rows_t, GaussianResponse(k=1.5))
print("\n[3%% tail] full-range C = %.4f +- %.4f   core C = %.4f +- %.4f   truth %.4f"
      % (gt["C"], gt.error("C"), gtk["C"], gtk.error("C"), T["C"]))
print("\nall closure checks passed")
