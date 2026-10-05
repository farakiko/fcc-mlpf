#!/usr/bin/env bash
# IDEA o2 production job: Pythia -> ddsim (full sim INCL. calo) -> digi+reco -> EOS.
# One file serves [T] + [C] + [PF]-prep (tracker digis+links, dual-readout calo digis+truth,
# truth tracks with AtCalorimeter state, classical ECAL clusters). Based on the colleague's
# IDEA_o2_v01 instructions (2026-10-05); configs vendored in cfg/.
#
# Usage:   ./gen_sim_digireco.sh <job_id> <n_events>
# Config via environment:
#   STAGING_DIR : dir containing this production/idea_o2/ content (cards, cfg, pythia.py)
#   OUTPUT_DIR  : EOS dir for <job_id>.edm4hep.root
#   K4H_NIGHTLY : nightlies date to pin, e.g. 2026-10-01  (REQUIRED -- unpinned nightlies
#                 are not reproducible; record what you used)
set -e
t1=$(date +%s)
if [[ $# -ne 2 ]]; then echo "Usage: $0 <job_id> <n_events>"; exit 1; fi
job_id=$1
n_events=$2
: "${STAGING_DIR:?set STAGING_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${K4H_NIGHTLY:?set K4H_NIGHTLY (nightlies date, e.g. 2026-10-01)}"

output_file="${OUTPUT_DIR}/${job_id}.edm4hep.root"
if [ -f "$output_file" ]; then echo "Output exists, skipping."; exit 0; fi
echo "job_id=${job_id} n_events=${n_events} nightly=${K4H_NIGHTLY}"

setup_key4hep() { source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r ${K4H_NIGHTLY}; }
ORIG_PARAMS=("$@"); set --; setup_key4hep; set -- "${ORIG_PARAMS[@]}"

xrdcp -s  "${STAGING_DIR}/cards/p8_ee_Zqq_ecm91.cmd" card.cmd
xrdcp -s  "${STAGING_DIR}/pythia.py" .
# Configure generator in-file rather than via CLI: recent nightlies ignore custom Gaudi
# instance names in option parsing (--Dumper.Filename was rejected; the writer registers
# as HepMCFileWriter.*). File-level config is robust to that drift.
sed -i "s|/path/to/pythia/card.cmd|card.cmd|" pythia.py
printf '\nhepmc_writer.Filename = "gen.hepmc"\n' >> pythia.py
xrdcp -sr "${STAGING_DIR}/cfg" .

if [ -n "${SEED_BASE:-}" ]; then
  proc=${job_id##*.}
  seed=$((SEED_BASE + proc))
else
  seed=$((1 + (0x$(openssl rand -hex 4) % 900000000)))
fi
printf '\nRandom:setSeed=on\nRandom:seed=%s\n' "$seed" >> card.cmd
echo "pythia seed=${seed}"

k4run pythia.py -n ${n_events}

ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o2_v01/IDEA_o2_v01.xml \
  --steeringFile cfg/SteeringFile_IDEA_o2_v01.py \
  --inputFiles gen.hepmc \
  --numberOfEvents -1 \
  --random.seed ${seed} \
  --outputFile sim.edm4hep.root

k4run cfg/run_digi_reco_nocluster.py \
  --IOSvc.Input sim.edm4hep.root \
  --IOSvc.Output digi_reco.edm4hep.root

echo "job=${job_id} nightly=${K4H_NIGHTLY} geo=IDEA_o2_v01 card=p8_ee_Zqq_ecm91 seed=${seed} date=$(date -Is)" > provenance.txt
xrdcp -f provenance.txt "${OUTPUT_DIR}/${job_id}.provenance.txt"
xrdcp digi_reco.edm4hep.root "${output_file}"

t2=$(date +%s)
echo "Runtime: $((t2 - t1)) seconds"
