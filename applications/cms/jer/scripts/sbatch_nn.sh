#!/bin/bash
# Training of the jernn network as a Slurm job on a GPU.  Submit FROM THIS APPLICATION DIRECTORY:
#
#     sbatch scripts/sbatch_nn.sh --files "/scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz" \
#         --cond genpt:log receta:abs --fraction 0.2 --outdir fig/nn_pt_eta
#     sbatch scripts/sbatch_nn.sh --files fig/nn_toy/toy.npz --weight none --cut --outdir fig/nn_toy/net
#
# All arguments go to train_nn.py.  Rows are held in memory: about 28 bytes per row with two
# conditions (the whole UL18 export, 519 M rows, needs ~15 GB plus copies; start with --fraction).
# The GPU partition name below is the CLIP one ("g"); change it if the cluster differs.
# The inputs are only read.  The log is jer-nn-<jobid>.log in the submit directory.
#SBATCH --job-name=jer-nn
#SBATCH --partition=g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=jer-nn-%j.log
. /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh && conda activate hephy-ml-gpu-claude
cd "${SLURM_SUBMIT_DIR:-.}"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "train_nn.py $@"; date
time python3 -u scripts/train_nn.py "$@"
echo FIT_DONE; date
