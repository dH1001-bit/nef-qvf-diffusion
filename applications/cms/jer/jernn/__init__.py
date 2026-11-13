"""jernn: a neural conditional density for the jet response (and later any target).

    features   variables as text specs (``genpt:log``, ``receta:abs``, ``rho``); derived columns; cuts
    data       one unbinned pass over npz files; unit weights when there is no weight column
    model      BornNet (torch): conditions -> (mu, sigma, c), exact normalised density, censoring;
               Predictor: a trained checkpoint, numpy in and out (params, log_prob, quantiles, PIT,
               sampling)
    diffusion  numpy only: the exact Ornstein-Uhlenbeck noising of a Born density (pdf and score at
               any diffusion time), for using the network as the ground truth of a diffusion model

``model`` needs torch; ``features``, ``data`` and ``diffusion`` need numpy only.
Scripts: scripts/train_nn.py, scripts/eval_nn.py, scripts/sbatch_nn.sh.
"""
