#!/bin/bash
# Calculation of the quantiles of the jernn network as a Slurm job on a GPU.
#
# All arguments go to quantiles_nn_global.py.  Rows are held in memory: about 28 bytes per row with two
# conditions (the whole UL18 export, 519 M rows, needs ~15 GB plus copies; start with --fraction).
# The GPU partition name below is the CLIP one ("g"); change it if the cluster differs.
# The inputs are only read.  The log is jer-nn-<jobid>.log in the submit directory.
#SBATCH --job-name=jer-q
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=jer-q-%j.log
. /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh && conda activate hephy-ml-gpu-claude
cd "${SLURM_SUBMIT_DIR:-.}"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "quantiles_nn_global.py $@"; date
time python3 -u quantiles_nn_global.py "$@"
echo FIT_DONE; date
