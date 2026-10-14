#!/bin/bash
# Submit one Slurm task per input file (never split a file list by fraction:
# that takes different fractions of each H_T slice and breaks the stitching).
#
#   submit_export.sh <era> <generator> <outdir> [extra export args...]
#   e.g.  submit_export.sh UL18 MadGraph /scratch-cbe/users/$USER/jer-unbinned/UL18
set -eu
ERA="$1"; GEN="$2"; OUT="$3"; shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"
M4="${M4_DIR:-/scratch-cbe/users/robert.schoefbeck/Jets/0070_FullRun2/M4_applyJEScorrections}"
mkdir -p "$OUT/logs"
LIST="$OUT/files_${ERA}_${GEN}.txt"
ls "$M4"/QCD_HT*_${GEN}_${ERA}/*.root | sort > "$LIST"
N=$(wc -l < "$LIST")
echo "$N files -> $OUT"
# TASKS=67,69,70  resubmits those task indices (same list order) instead of all
sbatch --array=${TASKS:-0-$((N-1))} --job-name=jer-export-$ERA --time=06:00:00 --mem=${MEM:-8G} --cpus-per-task=1 \
       --output="$OUT/logs/%a.log" --wrap="
f=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" '$LIST')
echo \"\$f\"
exec '$HERE/run_export.sh' \"\$f\" --out '$OUT' $*
"
