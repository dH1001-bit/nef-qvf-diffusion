#!/bin/bash
# A fit as a Slurm job.  Submit FROM THIS APPLICATION DIRECTORY:
#
#     sbatch scripts/sbatch_fit.sh <rows.npz> <outdir> [fit_response.py options]
#     sbatch --mem=48G --time=08:00:00 scripts/sbatch_fit.sh rows.npz fig/big --absy 0 0.5 ...
#
# Memory and time scale linearly with the rows: about 0.6 GB and 5 minutes
# per million rows on a cbe node (43 M rows: 25 GB, 3 h 17 min).  The log is
# jer-fit-<jobid>.log in the submit directory.
#SBATCH --job-name=jer-fit
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --output=jer-fit-%j.log
. /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh && conda activate hephy-ml-gpu-claude
cd "${SLURM_SUBMIT_DIR:-.}"
IN=$1; OUT=$2; shift 2
echo "rows $IN -> $OUT  ($@)"; date
time python3 -u scripts/fit_response.py "$IN" --out "$OUT" "$@"
echo FIT_DONE; date
