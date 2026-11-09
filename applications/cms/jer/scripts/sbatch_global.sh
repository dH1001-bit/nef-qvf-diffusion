#!/bin/bash
# The global (pt-smooth) amplitude fit as a Slurm job.  Submit FROM THIS APPLICATION DIRECTORY:
#
#     sbatch scripts/sbatch_global.sh                                   # read the export, fit all |eta| bins
#     sbatch scripts/sbatch_global.sh --load fig/global_receta/histograms2d.npz --K 2 2 2 2 1 1 1
#     sbatch scripts/sbatch_global.sh --load fig/global_receta/histograms2d.npz --eta-only 0 --outdir fig/g_eta0
#
# All arguments go to fit_response_global.py.  The first run reads the 72 export files once
# (one file in memory at a time) and writes histograms2d.npz; refits with --load skip that.
# The log is jer-global-<jobid>.log in the submit directory.
#SBATCH --job-name=jer-global
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=jer-global-%j.log
. /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh && conda activate hephy-ml-gpu-claude
cd "${SLURM_SUBMIT_DIR:-.}"
echo "fit_response_global.py $@"; date
time python3 -u scripts/fit_response_global.py "$@"
echo FIT_DONE; date
