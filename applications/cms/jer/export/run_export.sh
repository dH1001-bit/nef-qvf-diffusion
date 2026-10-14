#!/bin/bash
# Run the export for one input file inside the EL9 container with the
# Darwin/Core libraries.  Usage:  run_export.sh <M4 file> --out <dir> [args]
#
# Needs only: the CMS EL9 image and the LCG view on /cvmfs, and the Darwin/Core
# libraries, which the campaign copies into every sample directory next to the
# data (libDarwin.so, libCoreObjects.so, their .pcm and .rootmap).  Overrides:
#   EL9_SH    the container launcher          (default: el9.sh beside this script)
#   DAS_LIBS  directory with the libraries    (default: the input file's directory)
#   LCG_SETUP the LCG view                    (default: LCG_109 el9 gcc15)
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
INPUT="$1"
EL9_SH="${EL9_SH:-$HERE/el9.sh}"
DAS_LIBS="${DAS_LIBS:-$(cd "$(dirname "$INPUT")" && pwd)}"
LCG="${LCG_SETUP:-/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh}"
[ -r "$DAS_LIBS/libDarwin.so" ] || { echo "run_export.sh: no libDarwin.so in $DAS_LIBS (set DAS_LIBS)" >&2; exit 2; }
exec "$EL9_SH" bash -c "
source '$LCG'
export LD_LIBRARY_PATH='$DAS_LIBS':\$LD_LIBRARY_PATH
exec python3 '$HERE/export_response.py' $(printf '%q ' "$@")
"
