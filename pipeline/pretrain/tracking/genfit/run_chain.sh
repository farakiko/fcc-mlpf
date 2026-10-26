#!/bin/bash
# The ML -> Genfit chain (steps A-C) for several [T] models at once, over a list of source
# files of one sample, and optionally the fitted-level analysis (step E) of that sample.
# All models' tracks go into one file per source file and are fitted in one k4run pass
# (plus the detector's baseline tracks, refitted with Genfit), so the ROOT I/O, which
# dominates the cost, is paid once.
#
#   run_chain.sh -d cld -w work/cld_91gev -f files_91.txt \
#       -m "v1=runs/t_cld90k/best.pt" -m "v2=runs/t_cld_all/ep_019.pt" [-j 6] [-E] [-L "v1=..."]
#
#   -d DET      detector (genfit/detectors.py)
#   -w DIR      work directory of this sample: labels_<model>/, fitted/, logs/, analysis/
#   -f LIST     source files, one per line; for a detector whose step A reads the canonical
#               chunk (IDEA), each line is "<source file> <canonical chunk>"
#   -m N=CKPT   model name (used in collection names) and checkpoint; repeat
#   -L N=LABEL  plot label of model N (default: N)
#   -j JOBS     parallel fit processes (default 4)
#   -E          after the fits, run step E (analyze_fitted.py over all models, without the
#               baseline curve; compare_fit_failures.py) into <work>/analysis/
#   -B          step E also shows the Genfit-refitted baseline tracking
#
# Environment: TORCH_PY = python with torch/uproot/pyarrow (steps A, E; default python3);
# KEY4HEP_SETUP = shell command that sets up the key4hep nightly (steps B, C; default the
# pinned nightly of genfit/README.md).
#
# Every step is skip-if-exists, so the script can be rerun after any interruption. Several
# machines can work on the same work directory and file list at the same time: a file is
# claimed by creating <work>/claims/<file> (atomic mkdir). A claim left by a killed run is
# released by rerunning with -R (only when nothing else is running on that work dir).
set -o pipefail
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TORCH_PY=${TORCH_PY:-python3}
KEY4HEP_SETUP=${KEY4HEP_SETUP:-"source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r 2026-10-04"}
J=4; MODELS=(); LABELS=(); ANALYZE=0; WITH_BASE=0; RELEASE=0
while getopts "d:w:f:m:L:j:EBR" o; do
  case $o in
    d) DET=$OPTARG ;; w) W=$OPTARG ;; f) LIST=$OPTARG ;; m) MODELS+=("$OPTARG") ;;
    L) LABELS+=("$OPTARG") ;; j) J=$OPTARG ;; E) ANALYZE=1 ;; B) WITH_BASE=1 ;; R) RELEASE=1 ;;
    *) sed -n 2,30p "$0"; exit 1 ;;
  esac
done
[ -n "$DET" ] && [ -n "$W" ] && [ -f "$LIST" ] && [ ${#MODELS[@]} -gt 0 ] || { sed -n 2,30p "$0"; exit 1; }
mkdir -p "$W"/fitted "$W"/ml "$W"/logs "$W"/claims
W=$(cd "$W" && pwd)
[ $RELEASE = 1 ] && rm -rf "$W"/claims/* "$W"/fitted/*.tmp.edm4hep.root "$W"/ml/*
read -r STEP_A BASE < <($TORCH_PY -c "import sys; sys.path.insert(0, '$G'); import detectors as d; \
c = d.get('$DET'); print(c['step_a_input'], c['baseline_tracks'] or '-')") || exit 1
NAMES=(); for m in "${MODELS[@]}"; do NAMES+=("${m%%=*}"); done
label() { for l in "${LABELS[@]}"; do [ "${l%%=*}" = "$1" ] && { echo "${l#*=}"; return; }; done; echo "$1"; }

# A. inference, one process per model (GPU if present); step-A inputs per detector
if [ "$STEP_A" = canonical ]; then AIN=$(awk '{print $2}' "$LIST"); else AIN=$(awk '{print $1}' "$LIST"); fi
for m in "${MODELS[@]}"; do
  $TORCH_PY -u "$G"/export_infer.py --detector "$DET" --ckpt "${m#*=}" --files $AIN \
      --outdir "$W/labels_${m%%=*}" >> "$W/logs/A_${m%%=*}.log" 2>&1 || { echo "step A failed for ${m%%=*}"; exit 1; }
done
echo "$(date +%T) A done: ${#MODELS[@]} models x $(wc -l < "$LIST") files"

# B + C per file: all models' collections into one file, one fit pass for all of them
INS=""; OUTS=""; LABS=""
for n in "${NAMES[@]}"; do INS+="MLTracks_$n,"; OUTS+="GenfitTracks_$n,"; done
[ "$BASE" != - ] && { INS+="$BASE,"; OUTS+="GenfitBaseTracks,"; }
export G W DET KEY4HEP_SETUP INS=${INS%,} OUTS=${OUTS%,} NAMES_S="${NAMES[*]}"
awk '{print $1, ($2 == "" ? $1 : $2)}' "$LIST" | xargs -P "$J" -L 1 bash -c '
  set +u
  f=$0; b=$(basename "$f"); b=${b%.edm4hep.root}; out=$W/fitted/$b.fitted.edm4hep.root
  lb=$(basename "$1"); lb=${lb%.edm4hep.root}; lb=${lb%.parquet}     # step-A label file name
  [ -f "$out" ] && exit 0
  mkdir "$W/claims/$b" 2>/dev/null || exit 0                 # another process has it
  eval "$KEY4HEP_SETUP" >/dev/null 2>&1
  labs=""; for n in $NAMES_S; do labs+=" --labels MLTracks_$n=$W/labels_$n/$lb.labels.npz"; done
  t0=$(date +%s)
  python "$G"/write_mltracks.py --detector "$DET" --input "$f" --output "$W/ml/$b.ml.edm4hep.root" $labs \
    > "$W/logs/B_$b.log" 2>&1 || { echo "B FAILED $b"; rm -rf "$W/claims/$b"; exit 0; }
  k4run "$G"/run_genfit.py --detector "$DET" --inputFile "$W/ml/$b.ml.edm4hep.root" \
    --inputTracks "$INS" --outputTracks "$OUTS" --outputFile "$W/fitted/$b.tmp.edm4hep.root" \
    > "$W/logs/C_$b.log" 2>&1 || { echo "C FAILED $b"; rm -rf "$W/claims/$b"; exit 0; }
  mv "$W/fitted/$b.tmp.edm4hep.root" "$out" && rm -f "$W/ml/$b.ml.edm4hep.root"
  echo "$(date +%T) fitted $b ($(( $(date +%s) - t0 )) s)"'
NF=$(ls "$W"/fitted/*.fitted.edm4hep.root 2>/dev/null | wc -l)
echo "$(date +%T) B+C: $NF of $(wc -l < "$LIST") files fitted"
[ $ANALYZE = 1 ] || exit 0
[ "$NF" = "$(wc -l < "$LIST")" ] || { echo "not all files fitted (other machines still running?): step E skipped"; exit 0; }

# E. fitted-level analysis of this sample, one chain per model
tag=$(basename "$W"); A=$W/analysis; mkdir -p "$A"
chains=(); labs=()
for n in "${NAMES[@]}"; do
  l=$(label "$n"); labs+=("$l")
  chains+=(--chain "$l=$W/fitted/*.fitted.edm4hep.root@MLTracks_$n,GenfitTracks_$n")
done
if [ $WITH_BASE = 1 ] && [ "$BASE" != - ]; then
  chains+=(--chain "baseline tracking + Genfit=$W/fitted/*.fitted.edm4hep.root@$BASE,GenfitBaseTracks")
fi
$TORCH_PY -u "$G"/analyze_fitted.py --detector "$DET" --baseline none --tag "$tag" --outdir "$A" \
    "${chains[@]}" > "$A/analyze_$tag.log" 2>&1 || { echo "step E failed: $A/analyze_$tag.log"; exit 1; }
$TORCH_PY "$G"/compare_fit_failures.py --dir "$A" --tag "$tag" --chains "${labs[@]}" > /dev/null
echo "$(date +%T) E done -> $A"
