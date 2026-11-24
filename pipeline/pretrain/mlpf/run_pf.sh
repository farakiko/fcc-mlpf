#!/bin/bash
# [PF] end to end for one CLD sample: the [T] tracks fitted with Genfit2 (steps A-C of
# ../tracking/genfit/run_chain.sh, then step D), the [PF] shard cache (data.py) and the training
# (train.py). Every stage skips what exists, so the script can be rerun after an interruption or
# run stage by stage: the chain and the cache need CPUs only, the training a GPU.
#
# The fit writes a full copy of every event next to the fitted tracks (about 3 MB per tt̄ event),
# while [PF] needs only the per-track table of step D (about 7 kB per event). The chain therefore
# runs in batches of -b files: fit a batch, extract its track tables, delete its fitted files.
# What stays per source file is its step-A labels and its track table; a file that has a track
# table is never fitted again. -k keeps the fitted files (for the fitted-level analysis, step E).
#
#   run_pf.sh -s SOURCES -w WORK -m NAME=CKPT -o OUTDIR [-n NFILES] [-j JOBS] [-a NA -t TA] \
#       [-r K,P,F] [-b BATCH] [-k] [-g NGPU] [-S chain,cache,train] [-- <train.py options>]
#
#   -s SOURCES  EDM4hep REC source files of the sample, one per line
#   -w WORK     work directory: sources.txt, chain/ (labels_<N>/, tracks_<N>/), cache/ (shards)
#   -m N=CKPT   [T] model name and checkpoint (one model)
#   -o OUTDIR   training run directory
#   -n NFILES   use the first NFILES of the sorted sources (default all); the held-out set is
#               the first 10% of them
#   -j JOBS     parallel Genfit fits, step-D extractions and shard builders (default 8)
#   -a NA       run step A ([T] inference) in NA parallel processes of -t threads each
#               (default 1: run_chain.sh's single process; use NA > 1 when step A runs on CPU)
#   -r K,P,F    [T] slot acceptance with rescue (export_infer.py --rescue; default 4,0.95,0.05,
#               the CLD default of ../tracking/eval.py); labels made with another setting are
#               reused, so delete labels_<N>/ after changing it
#   -b BATCH    source files fitted per batch before their fitted files are deleted (default 200)
#   -k          keep the fitted files
#   -g NGPU     train on NGPU GPUs (torchrun, data parallel; default 1)
#   -S STAGES   comma-separated subset of chain,cache,train (default all three)
#
# Environment: TORCH_PY = python with torch/uproot/pyarrow (default python3); KEY4HEP_SETUP as
# in run_chain.sh. The batches release run_chain.sh's claims (-R), so one machine runs the chain
# of a work directory at a time.
set -eo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
G=$(cd "$H/../tracking/genfit" && pwd)
J=8; N=0; NA=1; TA=4; NG=1; B=200; RESCUE=4,0.95,0.05; KEEP=0; STAGES=chain,cache,train
while getopts "s:w:m:o:n:j:a:t:r:b:kg:S:" o; do
  case $o in
    s) SRC=$OPTARG ;; w) W=$OPTARG ;; m) M=$OPTARG ;; o) OUT=$OPTARG ;;
    n) N=$OPTARG ;; j) J=$OPTARG ;; a) NA=$OPTARG ;; t) TA=$OPTARG ;; r) RESCUE=$OPTARG ;; b) B=$OPTARG ;;
    k) KEEP=1 ;; g) NG=$OPTARG ;; S) STAGES=$OPTARG ;;
    *) sed -n 2,37p "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
[ "$1" = "--" ] && shift
[ -f "$SRC" ] && [ -n "$W" ] && [ -n "$M" ] || { sed -n 2,37p "$0"; exit 1; }
export TORCH_PY=${TORCH_PY:-python3}
NAME=${M%%=*}
mkdir -p "$W"
W=$(cd "$W" && pwd)
LIST=$W/sources.txt
sort "$SRC" | awk -v n="$N" 'n == 0 || NR <= n' > "$LIST"     # awk, not head: no SIGPIPE under pipefail
echo "$(date +%T) $(wc -l < "$LIST") source files, model $NAME, rescue $RESCUE, stages $STAGES"

# step D for one source file: the track table of its fitted file, then (unless -k) drop the
# fitted file; atomic, skip-if-exists
stepd() {
  local s; s=$(basename "$1"); s=${s%.edm4hep.root}
  local fit="$W/chain/fitted/$s.fitted.edm4hep.root" out="$W/chain/tracks_$NAME/$s.tracks.parquet"
  if [ ! -f "$out" ]; then
    if [ ! -f "$fit" ]; then echo "$s: no fitted file (fit failed?)"; return 0; fi
    if "$TORCH_PY" "$G/states_to_parquet.py" --input "$fit" --tracks "GenfitTracks_$NAME" \
        --output "$out.tmp.parquet" >> "$W/chain/logs/D_$NAME.log" 2>&1; then
      mv "$out.tmp.parquet" "$out"
    else
      echo "$s: step D failed (see logs/D_$NAME.log)"; return 0
    fi
  fi
  if [ "$KEEP" = 0 ]; then rm -f "$fit"; fi
  return 0
}
export -f stepd
export W NAME G KEEP

if [[ ",$STAGES," == *",chain,"* ]]; then
  mkdir -p "$W/chain/labels_$NAME" "$W/chain/tracks_$NAME" "$W/chain/logs"
  TODO=$W/chain/todo.txt
  while read -r f; do
    s=$(basename "$f"); [ -f "$W/chain/tracks_$NAME/${s%.edm4hep.root}.tracks.parquet" ] || echo "$f"
  done < "$LIST" > "$TODO"
  echo "$(date +%T) $(wc -l < "$TODO") files without a track table"
  if [ -s "$TODO" ] && [ "$NA" -gt 1 ]; then
    # step A in NA parallel processes (it skips files already labelled, so run_chain.sh's own
    # single-process step A then finds every file done); worth it when step A runs on CPU
    split -n "l/$NA" -d "$TODO" "$W/chain/stepA_part_"
    for p in "$W"/chain/stepA_part_*; do
      OMP_NUM_THREADS=$TA "$TORCH_PY" -u "$G/export_infer.py" --detector cld \
          --ckpt "${M#*=}" --rescue "$RESCUE" --files $(cat "$p") --outdir "$W/chain/labels_$NAME" \
          >> "$W/chain/logs/A_${NAME}_$(basename "$p").log" 2>&1 &
    done
    wait
    rm -f "$W"/chain/stepA_part_*
    echo "$(date +%T) parallel step A done ($NA processes)"
  fi
  if [ -s "$TODO" ]; then
    split -l "$B" -d -a 4 "$TODO" "$W/chain/batch_"
    for bf in "$W"/chain/batch_*; do
      echo "$(date +%T) batch $(basename "$bf"): $(wc -l < "$bf") files"
      bash "$G/run_chain.sh" -d cld -w "$W/chain" -f "$bf" -m "$M" -r "$RESCUE" -j "$J" -R
      xargs -P "$J" -I{} bash -c 'stepd "$1"' _ {} < "$bf"
      rm -f "$bf"
    done
  fi
  for f in $(cat "$LIST"); do stepd "$f"; done        # files fitted before this script deleted
  echo "$(date +%T) chain done: $(ls "$W/chain/tracks_$NAME" | grep -c '\.tracks\.parquet$') track tables, "\
"$(du -sh "$W/chain" | cut -f1) in chain/"
fi
cd "$H"
if [[ ",$STAGES," == *",cache,"* ]]; then
  "$TORCH_PY" -u data.py --sources "$LIST" --chain-work "$W/chain" --model "$NAME" \
      --shard-cache "$W/cache" --jobs "$J" --test-frac 0.1
fi
if [[ ",$STAGES," == *",train,"* ]]; then
  [ -n "$OUT" ] || { echo "-o OUTDIR is required for the train stage"; exit 1; }
  if [ "$NG" -gt 1 ]; then RUN=("$TORCH_PY" -m torch.distributed.run --standalone --nproc_per_node "$NG");
  else RUN=("$TORCH_PY" -u); fi
  "${RUN[@]}" train.py --sources "$LIST" --chain-work "$W/chain" --model "$NAME" \
      --shard-cache "$W/cache" --outdir "$OUT" "$@"
fi
