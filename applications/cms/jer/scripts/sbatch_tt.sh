#!/bin/bash
# The tensor-train amplitude fit (all |eta| bins jointly) as a Slurm job.  Submit FROM THIS
# APPLICATION DIRECTORY:
#
#     sbatch scripts/sbatch_tt.sh --load fig/global_receta/histograms2d.npz --init fig/global_K3/global_fits.json
#     sbatch scripts/sbatch_tt.sh --load ... --init ... --K 3 --chi 3 3 --outdir fig/tt_K3_chi3x3
#
# All arguments go to fit_response_tt.py.  Several bond dimensions: one job each, with --outdir.
# The inputs are only read.  The log is jer-tt-<jobid>.log in the submit directory.
#SBATCH --job-name=jer-tt
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=jer-tt-%j.log
. /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh && conda activate hephy-ml-gpu-claude
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
echo "fit_response_tt.py $@"; date
time python3 -u scripts/fit_response_tt.py "$@"
echo FIT_DONE; date
