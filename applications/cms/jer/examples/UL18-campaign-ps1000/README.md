# Example: the small test, campaign export, |y| < 0.5, prescale 1000

The standard test of the fit chain: 6 MB, 135 k rows, 45 seconds.

Input (not in git):

    cbe:  /scratch-cbe/users/robert.schoefbeck/jer-unbinned/campaign_UL18_absy0-0.5_ps1000.npz
    web:  https://schoef.web.cern.ch/schoef/nef-qvf-diffusion/JER/UL18-campaign-ps1000/campaign_UL18_absy0-0.5_ps1000.npz

made from the campaign's flat export (72 files, `jer_UL18_HT*.npz`) by

    python3 scripts/reduce_rows.py '/scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz' \
        campaign_UL18_absy0-0.5_ps10.npz --absy 0 0.5 --prescale 10
    python3 scripts/reduce_rows.py campaign_UL18_absy0-0.5_ps10.npz campaign_UL18_absy0-0.5_ps1000.npz --prescale 100

Command:

    python3 scripts/fit_response.py <path>/campaign_UL18_absy0-0.5_ps1000.npz --absy 0 0.5 --genpt 97 3500 \
        --edges 97 140 200 280 400 550 750 1000 1400 2000 3500 --bin 300 500 --out fig/ps1000

`--genpt 97` and the first edge at 97 are the campaign export's threshold.
There is no efficiency figure: that export carries no generator-jet table.

Result, 300 <= pt_gen < 500 GeV (28 k jets, n_eff 16 k):

    Gaussian, full range  : mu = 0.9978   sigma = 0.0702 +- 0.0006
    Gaussian, +-1.5s core : mu = 1.0012   sigma = 0.0619 +- 0.0009

against 0.0715 and 0.0624 at full statistics.  Files here: `fit.log`,
`results.json`, `response_bin`, `sigma_vs_pt_{full,core}`,
`pull_{full,core}` as pdf and png.  Published at
https://schoef.web.cern.ch/schoef/nef-qvf-diffusion/JER/UL18-campaign-ps1000/
