#!/bin/bash
# Run the export for one input file inside the EL9 container with the
# Darwin/Core libraries.  Usage:  run_export.sh <M4 file> --out <dir> [args]
#
# EL9_SH   : the container launcher (default: the campaign's scripts/el9.sh)
# DAS_LIBS : directory with libDarwin.so, libCoreObjects.so and their pcm/rootmap
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
EL9_SH="${EL9_SH:-$HOME/CMS/campaigns/0070/scripts/el9.sh}"
DAS_LIBS="${DAS_LIBS:-$HOME/CMS/DAS/tools/lib64}"
LCG="${LCG_SETUP:-/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh}"
exec "$EL9_SH" bash -c "
source '$LCG'
export LD_LIBRARY_PATH='$DAS_LIBS':\$LD_LIBRARY_PATH
exec python3 '$HERE/export_response.py' $(printf '%q ' "$@")
"
