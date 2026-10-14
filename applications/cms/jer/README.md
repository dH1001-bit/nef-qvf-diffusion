# jer: the unbinned JER fit

Lives in `applications/cms/jer/` of the nef-qvf-diffusion repository; on cbe
that is `~/ML/nef-qvf-diffusion/applications/cms/jer/`.  All commands below are
run from this directory.

An unbinned fit of the jet energy resolution (JER) on the simulated jet
response, with the NSC form

    sigma(pt)/pt = sqrt( sgn(N) N^2/pt^2 + S^2/pt + C^2 )

fitted directly on per-jet rows.  Gaussian response for now; the model
interface is where a non-Gaussian (NEF-QVF) response plugs in later.

## Dependencies, and the one Darwin-dependent file

The fit depends on **numpy and scipy only** (matplotlib for figures, optional).
No ROOT, no cvmfs, no container, no uproot.

The campaign n-tuples cannot be read without the Darwin framework: the branch
that marks a jet as rejected (`recJets.weights`, `jet.Weight() == 0`) has no
interpretation outside its dictionaries, and the friend-tree chaining
segfaults in a plain `TChain`.  All of that is confined to **one script**,
`export/export_response.py`, which runs once per input file inside the EL9
container (PyROOT + Darwin) and writes flat per-jet rows as `.npz`.  Nothing
else in this repository imports ROOT.

    export/export_response.py   PyROOT + Darwin.  One M4 file -> one npz.
    export/run_export.sh        container wrapper (LCG view, Darwin libs)
    export/submit_export.sh     Slurm array, one task per input file
    jerfit/                     the fit: numpy + scipy
    scripts/fit_response.py     command line: table, binned reference, unbinned fits, figures
    scripts/make_toy.py         synthetic rows in the same schema (develop without the cluster)
    tests/test_closure.py       closure of every fit on the synthetic sample

## Running

Export (on cbe; the handover documents why stage M4 and why one file at a time):

    export/run_export.sh /scratch-cbe/users/robert.schoefbeck/Jets/0070_FullRun2/M4_applyJEScorrections/QCD_HT300to500_MadGraph_UL18/0.root \
        --out /scratch-cbe/users/$USER/jer-unbinned/UL18 --max-events 20000      # a test
    export/submit_export.sh UL18 MadGraph /scratch-cbe/users/$USER/jer-unbinned/UL18   # all 72 files

Fit (anywhere with numpy/scipy):

    python3 tests/test_closure.py
    python3 scripts/make_toy.py out/toy --tail 0.03
    python3 scripts/fit_response.py 'out/toy/*.npz' --absy 0 2.5 --out out/toy-fig
    python3 scripts/fit_response.py 'rows/UL18/*.npz' --absy 0 0.5 --bin 300 500 --out fig/UL18

## The rows

See the docstring of `jerfit/rows.py`.  One row = one accepted reconstructed
jet paired with its nearest generator jet: `response`, `genpt`, `geny`,
`recpt`, `recy`, `weight` (= `genEvent.Weight() * jet.Weight()`, a cross
section in pb), `dr`, `dr2`.  The matching radius is an analysis cut
(`--max-dr`, default R/2 = 0.4), not an export choice: pairs are exported up to
0.8.  Each file also carries a generator-jet table (`g_pt`, `g_y`, `g_dr`) for
the matching efficiency, and the event count and summed event weight.

Weights are cross sections, so files and H_T slices simply concatenate.

## The fits, and two points of substance

* `fit_bin`: weighted Gaussian MLE in one bin, closed form.  `fit_core`: the
  campaign convention, a Gaussian truncated to mu +- k sigma, window iterated.
  The truncated likelihood is normalised on the fixed window of the previous
  iterate; a constant normalisation would make the iteration collapse.
* `fit_nsc`: the global unbinned fit, r | pt ~ N(mu, sigma_NSC(pt)), full range
  or windowed (per-row window mu +- k sigma(pt), iterated).
* `fit_per_bin` + `fit_nsc_to_points`: the binned reference.

**Errors.**  The likelihood is weighted, so the covariance is the sandwich
H^-1 J H^-1 with J = sum w_i^2 s_i s_i^T, not the inverse Hessian (which would
treat the summed cross section as a number of events).  `n_eff = (sum w)^2 /
sum w^2` is printed everywhere.

**Flattening.**  The cross-section weights fall like pt^-5, so a global
weighted likelihood is dominated by the lowest-pt rows and the constant term C
is unconstrained (the closure test shows errors ten times the binned ones).
Since the model is the conditional law p(r | pt), the weights may be multiplied
by any function of pt without biasing the estimator; `flatten` divides each
weight by the mean weight of its pt bin, keeping the relative weights at fixed
pt (the slice stitching and sample composition) and giving every pt region its
own statistical power.  The per-bin convention does the same thing implicitly.
The global fits in `fit_response.py` use flattened weights.

## The UL18 MadGraph export (2026-10-14)

All 72 M4 files of the UL18 MadGraph sample were exported on cbe (job arrays
3431457 and resubmissions; the files hold up to 6M events, not 1.8M, so the
export keeps its accumulators in typed arrays and needs 8 GB):

    /scratch-cbe/users/robert.schoefbeck/jer-unbinned/UL18/*.npz      72 files, 30 GB, 1.0e9 pairs
    /scratch-cbe/users/robert.schoefbeck/jer-unbinned/UL18_absy0-0.5.npz    |y| < 0.5, full statistics, 8.5 GB
    /scratch-cbe/users/robert.schoefbeck/jer-unbinned/UL18_student_ps50.npz all |y|, prescale 50, 745 MB

**For a student, start from the prescaled file**: 20M rows, fits on a laptop
in six minutes with `fit_response.py`, 570k jets in the 300-500 GeV, |y| < 0.5
bin.  A smaller one is one `reduce_rows.py --prescale 500` away.

Result on the prescaled set, 300 <= pt_gen < 500 GeV, |y| < 0.5 (`out/UL18-student-fig/`):

    Gaussian, full range  : mu = 0.9976   sigma = 0.0714 +- 0.0002
    Gaussian, +-1.5s core : mu = 1.0013   sigma = 0.0622 +- 0.0002
    unbinned NSC, full    : N = 3.73 +- 0.07   S = 1.111 +- 0.005   C = 0.0411 +- 0.0004
    unbinned NSC, core    : N ~ 0            S = 1.007 +- 0.005   C = 0.0317 +- 0.0004

The core width is the handover's 0.0622 to the last digit.  The full-range
width is not its 0.0643, and the difference is the method, not the sample: the
handover's "full range" was a chi^2 fit of a Gaussian to a histogram, which
the peak bins dominate, so it is a core fit in disguise; the unbinned Gaussian
MLE over the full range is the weighted RMS and takes the tails at face value.
That the two differ by 15% while the two core fits agree exactly is the
measurement of what the tails carry.  The binned chi^2/ndf of the NSC form to
the full-range widths is 165/11, to the core widths 43/11: neither set of
widths follows the NSC form within the statistics of this sample.

The matching efficiency (gen jet with an accepted rec jet within dR < 0.4)
plateaus at 97%, flat from 100 GeV to 2 TeV, not at 100%: the 3% are
generator jets whose reconstructed partner is rejected by the jet ID or the
veto map (the gen table measures the distance to the nearest *accepted* jet).
That population is removed, not smeared, and is not part of the response.

## What the toy shows

With a 3% low-side tail at response 0.7 (a stand-in for the out-of-cone
shoulder in the simulation, where the real distribution leaves a Gaussian by
two to three orders of magnitude), the full-range Gaussian returns C = 0.067
against a truth of 0.040 while the +-1.5 sigma core fit returns 0.040.  The
core width is not where the difficulty is; the tails are, and a tail-including
model is what the next step has to supply.

## Plugging in a different response model

`jerfit.nsc.fit_nsc` takes any object with `names`, `bounds`, `init(rows)`,
`window(theta, pt)` (None for the full range) and `logpdf(theta, r, pt, a, b)`
returning the per-row log density.  `GaussianResponse` is the reference
implementation.
