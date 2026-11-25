#!/bin/bash
# Rerun of the tail monotonicity penalty scan (lambda at J = 8) of sbatch_nn_scan.sh, now over the
# full OneCycle schedule.  In the first scan --patience 6 stopped every penalised run after 8-20 of
# the 40 epochs, before the learning rate was annealed (worth ~4e-4 per event in lnL), so they
# could not be compared with the unpenalised J8, which ran the whole schedule.  Here --patience 40
# (= --epochs) switches the early stop off; best.pt is still the best validation epoch.
# Submit FROM THE SCRIPTS DIRECTORY:
#
#     sbatch sbatch_nn_lam.sh             (all 3 runs; e.g. --array=0-1 for lambda 100 and 1000 only)
#
# ~220 s per epoch on one GPU, i.e. ~2.5 h per run.  The outdirs get the suffix _p40, so the
# early-stopped runs of the first scan stay untouched.  All runs use the rows of fig/nn_scan/J8
# (seed 0, --fraction 0.1), so they are compared row by row with J8 in ONE call, which also
# gives the paired statistical errors of the likelihood differences:
#
#     sbatch sbatch_quantiles_nn.sh --model lam0=fig/nn_scan/J8/best.pt \
#         lam100=fig/nn_scan/J8_lam100_p40/best.pt lam1000=fig/nn_scan/J8_lam1000_p40/best.pt \
#         lam10000=fig/nn_scan/J8_lam10000_p40/best.pt \
#         --global K3=<path>/global_K3/global_fits.json --outdir fig/nn_scan/cmp_mono_p40
#
# The inputs are only read.  Logs: jer-nn-lam-<jobid>_<index>.log in the submit directory.
#SBATCH --job-name=jer-nn-lam
#SBATCH --partition=g
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --output=jer-nn-lam-%A_%a.log
. /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh && conda activate hephy-ml-gpu-claude
cd "${SLURM_SUBMIT_DIR:-.}"

FILES="/scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz"
#      name              J  lambda
RUNS=("J8_lam100_p40     8  100"
      "J8_lam1000_p40    8  1000"
      "J8_lam10000_p40   8  10000")
read -r NAME J LAM <<< "${RUNS[${SLURM_ARRAY_TASK_ID:-0}]}"

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "run $NAME: J $J, lambda $LAM, patience 40"; date
time python3 -u train_nn.py --files "$FILES" --cond genpt:log receta:abs --fraction 0.1 \
    --J "$J" --mono-penalty "$LAM" --epochs 40 --patience 40 --outdir "fig/nn_scan/$NAME"
echo FIT_DONE; date
