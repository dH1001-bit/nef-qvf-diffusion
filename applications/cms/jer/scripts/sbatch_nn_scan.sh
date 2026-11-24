#!/bin/bash
# Scans of the jernn network as a Slurm job array on GPUs: the Hermite degree J (no penalty), and
# the tail monotonicity penalty lambda at J = 10.  Submit FROM THE SCRIPTS DIRECTORY:
#
#     sbatch sbatch_nn_scan.sh            (all 8 runs; or a subset, e.g. --array=0-4)
#
# Every run uses the settings of fig/nn_pt_eta (10 % of the UL18 export, conditions ln pt_gen and
# |receta|, seed 0): all runs see the same training and validation rows, so they can be compared
# row by row afterwards (quantiles_nn_global.py with several --model):
#
#     sbatch sbatch_quantiles_nn.sh --model J4=fig/nn_scan/J4/best.pt J6=fig/nn_scan/J6/best.pt \
#         J8=fig/nn_scan/J8/best.pt J10=fig/nn_scan/J10/best.pt J12=fig/nn_scan/J12/best.pt \
#         --global K3=<path>/global_K3/global_fits.json --outdir fig/nn_scan/cmp_J
#     sbatch sbatch_quantiles_nn.sh --model lam0=fig/nn_scan/J10/best.pt lam100=fig/nn_scan/J10_lam100/best.pt \
#         lam1000=fig/nn_scan/J10_lam1000/best.pt lam10000=fig/nn_scan/J10_lam10000/best.pt \
#         --outdir fig/nn_scan/cmp_mono
#
# lambda is in units of n_eff*lnL of the training rows (n_eff ~ 3.5e6 here); every run prints the
# penalty on fixed validation conditions ("mono(val)"), also the lambda = 0 runs: lambda * mono(val)
# of the unpenalised run is the scale where the penalty starts to cost likelihood.
# The inputs are only read.  Logs: jer-nn-scan-<jobid>_<index>.log in the submit directory.
#SBATCH --job-name=jer-nn-scan
#SBATCH --partition=g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7%2
#SBATCH --output=jer-nn-scan-%A_%a.log
. /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh && conda activate hephy-ml-gpu-claude
cd "${SLURM_SUBMIT_DIR:-.}"

FILES="/scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz"
#      name          J   lambda
RUNS=("J4            4   0"
      "J6            6   0"
      "J8            8   0"
      "J10           10  0"
      "J12           12  0"
      "J10_lam100    8  100"
      "J10_lam1000   8  1000"
      "J10_lam10000  8  10000")
read -r NAME J LAM <<< "${RUNS[${SLURM_ARRAY_TASK_ID:-0}]}"

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "run $NAME: J $J, lambda $LAM"; date
time python3 -u train_nn.py --files "$FILES" --cond genpt:log receta:abs --fraction 0.1 \
    --J "$J" --mono-penalty "$LAM" --outdir "fig/nn_scan/$NAME"
echo FIT_DONE; date
