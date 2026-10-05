#!/usr/bin/env bash
# Condor runner for batch_convert.py: one condor job = one stripe of chunks.
# Usage (set by run_pp_*.sub): pp_convert.sh <adapter> <job_index> <num_jobs>
# Env (from the sub file): REPO (fcc-mlpf clone on afs/eos), INPUTS (glob), OUTDIR,
#                          STACK_LABEL, FILES_PER_CHUNK, [MAX_FILES], [K4H_NIGHTLY]
set -e
adapter=$1; job_index=$2; num_jobs=$3
: "${REPO:?}"; : "${INPUTS:?}"; : "${OUTDIR:?}"; : "${STACK_LABEL:?}"; : "${FILES_PER_CHUNK:?}"
setup_env() {
  if [ "$adapter" = "idea" ]; then
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r "${K4H_NIGHTLY:?idea adapter needs K4H_NIGHTLY}"
  else
    source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
  fi
}
ORIG=("$@"); set --; setup_env; set -- "${ORIG[@]}"
python "$REPO/postprocessing/batch_convert.py" \
  --adapter "$adapter" --inputs "$INPUTS" --outdir "$OUTDIR" \
  --files-per-chunk "$FILES_PER_CHUNK" --nproc 1 \
  --stack "$STACK_LABEL" ${MAX_FILES:+--max-files $MAX_FILES} \
  --job-index "$job_index" --num-jobs "$num_jobs"
