#!/usr/bin/env bash
# Stage production inputs to EOS so condor jobs can xrdcp them (run from production/cld/).
# Usage: ./stage_to_eos.sh <STAGING_DIR>
set -e
DEST=${1:?usage: ./stage_to_eos.sh <STAGING_DIR on EOS>}
mkdir -p "$DEST"
cp -r cards pythia.py CLDConfig CLD_o2_v08 "$DEST/"
echo "staged to $DEST:"
ls "$DEST"
