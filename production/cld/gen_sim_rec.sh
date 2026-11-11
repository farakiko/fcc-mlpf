#!/usr/bin/env bash
# CLD production job: Pythia -> ddsim -> CLDReconstruction -> EOS, satisfying validation/README.md.
# Adapted (parameterized) from B. Dudar's gen_sim_rec.sh (verbatim copy in the initial git commit).
#
# Usage:   ./gen_sim_rec.sh <job_id> <n_events>
# Config:  via environment (defaults below) -- set STAGING_DIR and OUTPUT_DIR before condor use.
#   STAGING_DIR : EOS/afs dir containing this repo's production/cld/ (cards, CLDConfig, geometry)
#   OUTPUT_DIR  : EOS dir where <job_id>.edm4hep.root lands
#   K4H_RELEASE : pinned key4hep release (DO NOT float to nightlies for production)
#   GEOMETRY    : compact XML, relative to STAGING_DIR
#   CARD        : pythia card, relative to STAGING_DIR
#   SEED_BASE   : seed of process 0 (seed = SEED_BASE + process index), or "random" for a
#                 smoke test; see campaigns/CAMPAIGNS.md before choosing a block
set -e
t1=$(date +%s)

if [[ $# -ne 2 ]]; then echo "Usage: $0 <job_id> <n_events>"; exit 1; fi
job_id=$1
n_events=$2

: "${STAGING_DIR:?set STAGING_DIR (e.g. /eos/user/<u>/<you>/fcc-mlpf/production/cld)}"
: "${OUTPUT_DIR:?set OUTPUT_DIR (e.g. /eos/user/<u>/<you>/cld_smoke)}"
K4H_RELEASE=${K4H_RELEASE:-2026-04-08}
GEOMETRY=${GEOMETRY:-CLD_o2_v08/CLD_o2_v08.xml}
CARD=${CARD:-cards/p8_ee_tt_ecm365.cmd}

output_file="${OUTPUT_DIR}/${job_id}.edm4hep.root"
if [ -f "$output_file" ]; then
  echo "Output exists, skipping (rerun after eviction)."; exit 0
fi
echo "job_id=${job_id} n_events=${n_events} release=${K4H_RELEASE} geo=${GEOMETRY} card=${CARD}"

# wrapper avoids leaking CLI args into the sourced setup
setup_key4hep() { source /cvmfs/sw.hsf.org/key4hep/setup.sh -r ${K4H_RELEASE}; }
setup_key4hep

# stage inputs to the (condor scratch) cwd
xrdcp -s  "${STAGING_DIR}/${CARD}" card.cmd
xrdcp -s  "${STAGING_DIR}/pythia.py" .
xrdcp -sr "${STAGING_DIR}/$(dirname ${GEOMETRY})" .
xrdcp -sr "${STAGING_DIR}/CLDConfig" .

# Per-job seed: seed = SEED_BASE + process index, so a campaign occupies one contiguous block
# that can be checked against campaigns/*.seeds.txt before submission (CAMPAIGNS.md). A
# random seed must be asked for explicitly (SEED_BASE=random, smoke tests only).
: "${SEED_BASE:?set SEED_BASE (campaigns/CAMPAIGNS.md) or SEED_BASE=random for a smoke test}"
if [ "${SEED_BASE}" = random ]; then
  seed=$((1 + (0x$(openssl rand -hex 4) % 900000000)))
else
  proc=${job_id##*.}                          # job_id = Cluster.Process
  seed=$((SEED_BASE + proc))
fi
sed -i "s/^Random:seed.*/Random:seed = ${seed}/" card.cmd
echo "pythia seed=${seed}"

k4run pythia.py \
  --Pythia8.PythiaInterface.pythiacard card.cmd \
  --Dumper.Filename gen.hepmc \
  --num-events ${n_events}

ddsim \
  --steeringFile CLDConfig/cld_steer.py \
  --compactFile "${GEOMETRY}" \
  --inputFiles gen.hepmc \
  --outputFile sim.edm4hep.root \
  --numberOfEvents -1

cd CLDConfig
k4run CLDReconstruction.py \
  --compactFile "../${GEOMETRY}" \
  --inputFiles ../sim.edm4hep.root \
  --outputBasename ../rec \
  --num-events -1
cd ..

# provenance sidecar (one line, greppable)
echo "job=${job_id} release=${K4H_RELEASE} geo=${GEOMETRY} card=${CARD} seed=${seed} date=$(date -Is)" \
  > provenance.txt
xrdcp -f provenance.txt "${OUTPUT_DIR}/${job_id}.provenance.txt"
xrdcp rec_REC.edm4hep.root "${output_file}"

t2=$(date +%s)
echo "Runtime: $((t2 - t1)) seconds"
