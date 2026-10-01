#!/usr/bin/env bash
set -e

t1=$(date +%s)

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <job_id> <n_events>"
  exit 1
fi

job_id=$1
n_events=$2

project_dir="/eos/home-b/bdudar/1-projects/farouk_tt_cld_tracking"
output_file="${project_dir}/data/${job_id}.edm4hep.root"

if [ -f "$output_file" ]; then
  echo "Output file already exists, skipping (likely a rerun after eviction)."
  exit 0
fi

echo "job_id ${job_id}"
echo "n_events ${n_events}"

# Wrapper function avoids passing CLI arguments to the source command
setup_key4hep_environment() {
  source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08
}
setup_key4hep_environment

# Temporary fix for nighlies after 2026-09-29 when the new Gaudi broke things
# export GAUDI_PLUGIN_PATH=$LD_LIBRARY_PATH


xrdcp -s ${project_dir}/p8_ee_tt_ecm365.cmd .
xrdcp -s ${project_dir}/pythia.py .
xrdcp -sr ${project_dir}/CLD_o2_v08 .
xrdcp -sr ${project_dir}/CLDConfig .

seed=$((1 + (0x$(openssl rand -hex 4) % 900000000))) # generate random seeds on a fly
sed -i "s/^Random:seed.*/Random:seed = ${seed}/" "p8_ee_tt_ecm365.cmd"

k4run pythia.py \
  --Pythia8.PythiaInterface.pythiacard "p8_ee_tt_ecm365.cmd" \
  --Dumper.Filename "gen.hepmc" \
  --num-events ${n_events}

ddsim \
  --steeringFile "CLDConfig/cld_steer.py" \
  --compactFile "CLD_o2_v08/CLD_o2_v08.xml" \
  --inputFiles "gen.hepmc" \
  --outputFile "sim.edm4hep.root" \
  --numberOfEvents -1

cd ./CLDConfig
k4run CLDReconstruction.py \
  --compactFile "../CLD_o2_v08/CLD_o2_v08.xml" \
  --inputFiles "../sim.edm4hep.root" \
  --outputBasename "../rec" \
  --num-events -1
cd ..

xrdcp rec_REC.edm4hep.root ${output_file}

t2=$(date +%s)
echo "Runtime: $((t2 - t1)) seconds"
