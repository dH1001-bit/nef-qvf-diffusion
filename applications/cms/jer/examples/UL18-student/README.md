# Example: UL18 MadGraph, prescaled student set, |y| < 0.5

Input (not in git, 745 MB; made with the export in `export/` before it
applied the pile-up-cleaning event veto, so it contains the 0.5-2.6% of
events that `applyPUcleaning` vetoes.  The campaign's own flat export, the
student's starting point, applies the veto; see the top-level README):

    cbe:    /scratch-cbe/users/robert.schoefbeck/jer-unbinned/UL18_student_ps50.npz
    laptop: ~/CMS/jer-unbinned/out/UL18/UL18_student_ps50.npz (or wherever you copy it)

produced from the 72 exported files by

    python3 scripts/reduce_rows.py '/scratch-cbe/users/robert.schoefbeck/jer-unbinned/UL18/QCD_*.npz' \
        UL18_student_ps50.npz --prescale 50

Command (six minutes on a laptop):

    python3 scripts/fit_response.py <path>/UL18_student_ps50.npz --absy 0 0.5 --bin 300 500 --out out/UL18-student-fig

Files here: `fit.log` (the printed output), `results.json` (all fitted
values, errors, per-bin efficiency), `response_bin.pdf` (the 300-500 GeV bin
with the two Gaussians), `sigma_vs_pt_{full,core}.pdf` (per-bin widths and
the NSC curves), `pull_{full,core}.pdf` (standardised residual per pt bin),
`efficiency.pdf` (matching efficiency).
