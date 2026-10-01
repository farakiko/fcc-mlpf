#!/usr/bin/env bash
# Stage production inputs to EOS so condor jobs can xrdcp them (run from production/cld/).
# Usage: ./stage_to_eos.sh [STAGING_DIR]
set -e
DEST=${1:-/eos/user/f/fmokhtar/fcc-mlpf/production/cld}
mkdir -p "$DEST"
cp -r cards pythia.py CLDConfig CLD_o2_v08 gen_sim_rec.sh "$DEST/"
echo "staged to $DEST:"
ls "$DEST"
