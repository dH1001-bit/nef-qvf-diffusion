# jer: the unbinned JER fit

Lives in `applications/cms/jer/` of the nef-qvf-diffusion repository; on cbe
that is `~/ML/nef-qvf-diffusion/applications/cms/jer/`.  All commands below are
run from this directory.

An unbinned fit of the jet energy resolution (JER) on the simulated jet
response, with the NSC form

    sigma(pt)/pt = sqrt( sgn(N) N^2/pt^2 + S^2/pt + C^2 )

fitted directly on per-jet rows.  Gaussian response for now; the model
interface is where a non-Gaussian (NEF-QVF) response plugs in later.

## Dependencies: numpy and scipy, and no EOS

The fit depends on **numpy and scipy only** (matplotlib for figures, optional).
No ROOT, no cvmfs, no container, no uproot, no grid proxy.

The data come as flat per-jet arrays on cbe scratch, readable with numpy
alone.  Two exports exist; both were produced once by someone with EOS access
and nothing downstream touches EOS, Darwin or the campaign repository again.

**The student starts from the campaign's export** (campaign 0070,
`plotting/analysis/study/export_jer_rows.py`, documented in its
`docs/jer_unbinned_handover.md`):

    /scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT<slice>_<k>.npz
        72 files, 15.8 GB, 519 M rows from 380 M events, world-readable

The export in this directory (`export/`) is the one Darwin-dependent piece of
code here.  It exists because the fit wants three things the campaign export
does not carry: pairs below 97 GeV (the N term lives there), pairs out to
dR < 0.8 with the second-nearest distance (the matching radius as an analysis
cut, isolation as a systematic), and a generator-jet table (the matching
efficiency).  Like any re-export it reads EOS through the n-tuples' friend
chain and needs a grid proxy; see "The exports" below.

    export/export_response.py   PyROOT + Darwin.  One M4 file -> one npz.  Needs EOS.
    export/run_export.sh        container wrapper (LCG view, Darwin libs from the sample directory)
    export/el9.sh               the EL9 container launcher (copied from campaign 0070)
    export/submit_export.sh     Slurm array, one task per input file
    jerfit/                     the fit: numpy + scipy
    scripts/fit_response.py     command line: table, binned reference, unbinned fits, figures
    scripts/reduce_rows.py      cuts and prescale, file by file, into one fit-sized npz
    scripts/make_toy.py         synthetic rows in the same schema (develop without the cluster)
    tests/test_closure.py       closure of every fit on the synthetic sample

## Running

**The smallest real test** (seconds; this is the file to start with):

    python3 scripts/fit_response.py /scratch-cbe/users/robert.schoefbeck/jer-unbinned/campaign_UL18_absy0-0.5_ps1000.npz \
        --absy 0 0.5 --genpt 97 3500 --edges 97 140 200 280 400 550 750 1000 1400 2000 3500 --bin 300 500 --out fig/ps1000

6 MB, 135 k rows (|y| < 0.5, prescale 1000, n_eff 9 k), 28 k jets in the
300-500 GeV bin.  The file is also on the web, so the test runs anywhere
with numpy and scipy (`examples/UL18-campaign-ps1000/` has the output):

    https://schoef.web.cern.ch/schoef/nef-qvf-diffusion/JER/UL18-campaign-ps1000/campaign_UL18_absy0-0.5_ps1000.npz
  Every step of the chain runs on it in
under a minute, and the 300-500 GeV widths come out 0.0619 +- 0.0009 (core)
and 0.0702 +- 0.0006 (full range), within errors of the full-statistics
values below.  Sizes in between are one `reduce_rows.py --prescale` away
(`campaign_UL18_absy0-0.5_ps10.npz`, 558 MB, is the same selection at
prescale 10; it fits in about an hour on a cluster node).

On the campaign export itself (cbe, any Python with numpy/scipy, e.g. the
conda environment `hephy-ml-gpu-claude`):

    python3 scripts/reduce_rows.py '/scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz' \
        my_UL18_absy0-0.5_ps10.npz --absy 0 0.5 --prescale 10
    python3 scripts/fit_response.py my_UL18_absy0-0.5_ps10.npz --absy 0 0.5 --genpt 97 3500 \
        --edges 97 140 200 280 400 550 750 1000 1400 2000 3500 --bin 300 500 --out fig/campaign

`--genpt 97` and the edges starting at 97 match that export's threshold; the
defaults (30 GeV, edges from 30) are for the export in this directory.

Without the cluster:

    python3 tests/test_closure.py
    python3 scripts/make_toy.py out/toy --tail 0.03
    python3 scripts/fit_response.py 'out/toy/*.npz' --absy 0 2.5 --out out/toy-fig

Re-exporting with `export/` (cbe, needs a grid proxy):

    export/run_export.sh /scratch-cbe/users/robert.schoefbeck/Jets/0070_FullRun2/M4_applyJEScorrections/QCD_HT300to500_MadGraph_UL18/0.root \
        --out /scratch-cbe/users/$USER/jer-unbinned/UL18 --max-events 20000      # a test
    export/submit_export.sh UL18 MadGraph /scratch-cbe/users/$USER/jer-unbinned/UL18   # all 72 files

## The rows

`jerfit.rows.load` reads both exports into one schema (docstring of
`jerfit/rows.py`).  One row = one accepted reconstructed jet paired with its
nearest generator jet: `response`, `genpt`, `geny`, `recpt`, `recy`,
`weight` (a cross section in pb, all factors applied), `dr`.  Weights are cross
sections, so files and H_T slices simply concatenate.

|                        | campaign export (`jer_UL18_HT*.npz`) | this export (`export/`)              |
|------------------------|--------------------------------------|--------------------------------------|
| generator-jet pt       | > 97 GeV                             | > 20 GeV (`--genpt-min`)             |
| matching               | nearest gen jet within dR < 0.4      | nearest gen jet, kept to dR < 0.8; `dr2` to the second-nearest |
| generator-jet table    | no                                   | yes (`g_pt`, `g_y`, `g_dr`): matching efficiency |
| pile-up columns        | `rho`, `trpu`, `nVtx`, `area`, both eta | no                                |
| event veto (PU cleaning, `recEvent.Weight()==0`) | applied | applied since 2026-10-14 evening; the UL18 files below predate it |
| per file               | `n_events_read`, `n_events_vetoed`, `sum_weights` (read as `n_events`, `sumw_events`) | `n_events`, `sumw_events`, jet counts |

Two properties of either export the fit must respect (both from the handover,
both built in):

* **The per-row weight spans six orders of magnitude** between the lightest
  and the heaviest H_T slice, so 519 M rows carry nothing like 519 M rows'
  worth of statistics.  The likelihood is weighted and the errors come from
  sums of weight squared (the sandwich, below); `n_eff = (sum w)^2 / sum w^2`
  is printed everywhere.
* **The matching is not a bijection.**  Two reconstructed jets may claim the
  same generator jet, and the exported response reaches 34 where a hard jet
  attached itself to a soft neighbour.  `fit_response.py` restricts the
  response to `--response 0 2` by default.  Tightening the radius needs no
  re-export (`--max-dr 0.2`); demanding isolation (`--isolate`) needs `dr2`,
  i.e. the export in this directory.

The handover's 58% is the fraction of reconstructed jets that find a generator
partner; the pile-up and soft jets without one are not resolution.  The
quantity that matters for the response is the fraction of generator jets with
an accepted reconstructed partner, which this export's gen table gives: 97%,
flat from 100 GeV to 2 TeV (below).

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
treat the summed cross section as a number of events).

**Flattening.**  The cross-section weights fall like pt^-5, so a global
weighted likelihood is dominated by the lowest-pt rows and the constant term C
is unconstrained (the closure test shows errors ten times the binned ones).
Since the model is the conditional law p(r | pt), the weights may be multiplied
by any function of pt without biasing the estimator; `flatten` divides each
weight by the mean weight of its pt bin, keeping the relative weights at fixed
pt (the slice stitching and sample composition) and giving every pt region its
own statistical power.  The per-bin convention does the same thing implicitly.
The global fits in `fit_response.py` use flattened weights.

## The exports, and where EOS enters

Nothing from the campaign repository is needed to run `export/`.  The
container launcher is here, and the Darwin/Core libraries (`libDarwin.so`,
`libCoreObjects.so`, `.pcm`, `.rootmap`) are taken from the input file's own
directory, where the campaign copies them next to the data.  Beyond that:
`/cvmfs` (the CMS EL9 image and the LCG view), read access to the M4 stage on
`/scratch-cbe`, and **a grid proxy** (`~/.private/.proxy`, or
`X509_USER_PROXY`).

The proxy is needed because of where the generator jets live.  Every stage
file on scratch holds only its own branch and reaches the others through a
chain of friend trees: M4 -> M3 -> M2 (`genEvent`) -> M1, all on scratch, and
M1's friend for `genJets` (and `pileup`) is the raw CRAB output on EOS
(`root://eos.grid.vbc.ac.at//eos/vbc/experiments/cms/store/user/schoef/FullRun2_AK8/...`,
99 GB for one H_T slice, 5 TB in all, readable only with CMS group
membership).  Without the proxy the friend silently fails to open and the
tree has no `genJets`.  This is the same for the campaign's export and for
this one, and it is why an export pass exists at all: paid once, by someone
who has the access.

The UL18 MadGraph export made with `export/` on 2026-10-14 (72 M4 files, job
arrays 3431457 and resubmissions; the files hold up to 6 M events, so the
accumulators are typed arrays and the tasks need 8 GB):

    /scratch-cbe/users/robert.schoefbeck/jer-unbinned/UL18/*.npz             72 files, 30 GB, 1.0e9 pairs
    /scratch-cbe/users/robert.schoefbeck/jer-unbinned/UL18_absy0-0.5.npz     |y| < 0.5, full statistics, 8.5 GB
    /scratch-cbe/users/robert.schoefbeck/jer-unbinned/UL18_absy0-0.5_ps5.npz |y| < 0.5, prescale 5, 1.7 GB
    /scratch-cbe/users/robert.schoefbeck/jer-unbinned/UL18_student_ps50.npz  all |y|, prescale 50, 745 MB

These predate the event-veto fix and contain the 0.5-2.6% of events that
`applyPUcleaning` vetoes.  Their effect is measured by the comparison with the
campaign export below.

## Results, UL18 MadGraph, |y| < 0.5

Prescale-50 set from this export, 300 <= pt_gen < 500 GeV, 570 k jets
(`examples/UL18-student/`, published at
https://schoef.web.cern.ch/schoef/nef-qvf-diffusion/JER/UL18-student/):

    Gaussian, full range  : mu = 0.9976   sigma = 0.0714 +- 0.0002
    Gaussian, +-1.5s core : mu = 1.0013   sigma = 0.0622 +- 0.0002
    unbinned NSC, full    : N = 3.73 +- 0.07   S = 1.111 +- 0.005   C = 0.0411 +- 0.0004
    unbinned NSC, core    : N ~ 0            S = 1.007 +- 0.005   C = 0.0317 +- 0.0004

Prescale-5 set, 5.7 M jets in the bin (job 3433121, 3 h 17 min, 25 GB; the
fitter uses numerical gradients and scales linearly with the rows, so the
full-statistics file is a question of patience, not of memory):

    Gaussian, full range  : mu = 0.9977   sigma = 0.0715 +- 0.0001
    Gaussian, +-1.5s core : mu = 1.0012   sigma = 0.0624 +- 0.0001
    unbinned NSC, full    : N = 3.68 +- 0.02   S = 1.115 +- 0.002   C = 0.0408 +- 0.0001
    unbinned NSC, core    : N ~ 0            S = 1.010 +- 0.001   C = 0.0315 +- 0.0001
    binned chi2/ndf       : 1593/11 (full-range widths)   225/11 (core widths)

The core width is the handover's 0.0622 to the last digit.  The full-range
width is not its 0.0643, and the difference is the method, not the sample: the
handover's "full range" was a chi^2 fit of a Gaussian to a histogram, which
the peak bins dominate, so it is a core fit in disguise; the unbinned Gaussian
MLE over the full range is the weighted RMS and takes the tails at face value.
That the two differ by 15% while the two core fits agree exactly is the
measurement of what the tails carry.  With five times the statistics the NSC
form fails the per-bin widths five times harder: neither the full-range nor
the core widths follow it.

The matching efficiency (gen jet with an accepted rec jet within dR < 0.4)
plateaus at 97%, flat from 100 GeV to 2 TeV, not at 100%: the 3% are
generator jets whose reconstructed partner is rejected by the jet ID or the
veto map (the gen table measures the distance to the nearest *accepted* jet).
That population is removed, not smeared, and is not part of the response.

**Cross-check on the campaign export** (|y| < 0.5, pt_gen > 97 GeV).
Prediction, recorded first: the 300-500 GeV core width 0.0622-0.0624, the
full-range width 0.0714-0.0715; the two exports differ only by the event veto,
which removes soft high-weight events (6% of the summed event weight for
0.5-2.6% of the events), so any shift should sit in the lowest pt bins.
Prescale 1000 (the test file above, 45 s; `examples/UL18-campaign-ps1000/`,
published at https://schoef.web.cern.ch/schoef/nef-qvf-diffusion/JER/UL18-campaign-ps1000/):
core 0.0619 +- 0.0009, full range 0.0702 +- 0.0006, both within two standard
errors of the prediction.  The
prescale-10 run (job 3442538, log `fit_campaign_absy0-0.5_ps10.log`,
figures `fig_campaign_absy0-0.5_ps10/` on scratch) is the precise version.

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
