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
    scripts/sbatch_fit.sh       the same fit as a Slurm job (anything above laptop size)
    scripts/make_toy.py         synthetic rows in the same schema (develop without the cluster)
    scripts/hermite_demo.py     the Hermite OPS and the Born model, step by step; the student's starting point for the tail fit
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

**From the small test to full statistics.**  The same command runs at
every size; only the input changes.  The per-bin Gaussian fits are closed
form and cost nothing, the global unbinned NSC fit uses numerical gradients
and scales linearly with the rows: about 6 minutes and 0.3 GB per million
rows on a cbe node (13.5 M rows: 80 min, 4.1 GB).  So:

| input (|y| < 0.5)                      | rows   | size   | fit            | where          |
|----------------------------------------|--------|--------|----------------|----------------|
| `campaign_UL18_absy0-0.5_ps1000.npz`   | 135 k  | 6 MB   | 45 s           | laptop         |
| `campaign_UL18_absy0-0.5_ps10.npz`     | 13.5 M | 558 MB | 80 min, 4 GB   | Slurm job      |
| campaign export, no prescale           | 135 M  | 5.6 GB | about 13 h, 40 GB | not yet practical |

Anything beyond the laptop goes through `scripts/sbatch_fit.sh`, submitted
from this directory (it activates the conda environment and writes
`jer-fit-<jobid>.log` here):

    sbatch scripts/sbatch_fit.sh /scratch-cbe/users/robert.schoefbeck/jer-unbinned/campaign_UL18_absy0-0.5_ps10.npz fig/ps10 \
        --absy 0 0.5 --genpt 97 3500 --edges 97 140 200 280 400 550 750 1000 1400 2000 3500 --bin 300 500
    sbatch --mem=48G scripts/sbatch_fit.sh <a bigger file> fig/big ...

Prescale 10 is full statistics for every practical purpose: its errors are
sqrt(10) times the full-sample ones (0.0001 instead of 0.00003 on the
300-500 GeV core width).  The unprescaled fit exceeds the 8 h limit; what
makes it affordable is an analytic gradient in `jerfit/mle.py`, not a
bigger node, and that is the next step on the fitter if full statistics
are ever needed.  Any size in between is one `reduce_rows.py --prescale`
call on the prescale-10 file (the prescale multiplies the weights, so sums
of weights are preserved in expectation and the files remain comparable).

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
errors of the prediction.  Prescale 10 (job 3442538, 80 min; `fig_campaign_absy0-0.5_ps10/` on
scratch, published at
https://schoef.web.cern.ch/schoef/nef-qvf-diffusion/JER/UL18-campaign-ps10/):
core 0.0623 +- 0.0001, as predicted; full range **0.0703 +- 0.0001**, not the
predicted 0.0714.  The prediction was wrong about where the veto acts.  Bin
by bin, the core widths of the two exports agree to 0.0002 over the whole
range, while the full-range widths separate with pt:

    pt [GeV]      322     453     620     838    1127    1571    2240
    this export  0.0741  0.0658  0.0608  0.0574  0.0556  0.0540  0.0543
    campaign     0.0732  0.0636  0.0574  0.0526  0.0494  0.0481  0.0464
    core, both   0.0649  0.0569  0.0509  0.0464  0.0430  0.0403  0.0386

The vetoed events are not soft; they are events in which a reconstructed jet
is far harder than the generator-level event (the "over-weighted
reconstruction artefacts" of the campaign's cleaning), so they populate the
high-side tail at high pt with the large weights of the low-H_T slices.  The
full-range width at 1 TeV depends on that veto at the 12% level, the core
width not at all.  This is the cleanest demonstration so far that the
full-range Gaussian width measures the tails and their definition, and that a
tail model is needed before any full-range number means anything.  All
further work uses the campaign export, where the veto is applied.

## What the toy shows

With a 3% low-side tail at response 0.7 (a stand-in for the out-of-cone
shoulder in the simulation, where the real distribution leaves a Gaussian by
two to three orders of magnitude), the full-range Gaussian returns C = 0.067
against a truth of 0.040 while the +-1.5 sigma core fit returns 0.040.  The
core width is not where the difficulty is; the tails are, and a tail-including
model is what the next step has to supply.

## The non-Gaussian response: the Hermite demo and the student's task

`scripts/hermite_demo.py` is a walk-through, meant to be read line by line
(everything is in the module scope, so run it with `ipython -i` and inspect
the variables afterwards):

    . /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh && conda activate hephy-ml-gpu-claude
    cd ~/ML/nef-qvf-diffusion/applications/cms/jer
    ipython -i scripts/hermite_demo.py

It shows, with a numerical check at every step: the orthonormal Hermite basis
of N(mu, sigma^2) as `nefqvf` evaluates it (`Normal.basis`, `basis_dot`,
`jacobi_coefficients`); the Born model p = p_ref h^2 with h = sum c_n phi_n,
whose normalisation is the unit sphere ||c|| = 1 and whose moments
E_p[phi_k] = c^T Phi_k c come from the exact product tensor
(`product_matrices`, `ratio_coefficients`); a density with a low-side tail
built from degrees 3 and 4, a sample from it, and the weighted log-likelihood
of the model evaluated at the truth and at the Gaussian point; and finally the
JER rows of the 300-500 GeV bin, where the weighted moments <phi_3> and
<phi_4> of the response are measured at 4.5 and 3.7 standard errors from the
Gaussian's zero.  Those moments are what an amplitude has to reproduce.

The fit is deliberately not in the demo.  The task, as the demo prints it:
fit the Born model with active degrees {3, 4} and then {3, ..., 6} to the rows
by maximising the weighted likelihood on the sphere, with sandwich errors;
check the Gaussian limit, the moments pull by pull, the likelihood gain
against the number of degrees, and the histogram on a log scale; then let mu
and sigma follow the NSC form through the model interface below.

## Plugging in a different response model

`jerfit.nsc.fit_nsc` takes any object with `names`, `bounds`, `init(rows)`,
`window(theta, pt)` (None for the full range) and `logpdf(theta, r, pt, a, b)`
returning the per-row log density.  `GaussianResponse` is the reference
implementation.
