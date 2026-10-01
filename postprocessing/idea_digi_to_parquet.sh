#!/usr/bin/env bash
# IDEA one-step conversion: digi EDM4hep ROOT -> canonical training parquet.
# Internally: production/idea/src/process_tree.py (De Vita's converter, enriched; emits his
# graph format as an EPHEMERAL intermediate) -> postprocessing/idea.py (canonical schema).
# The intermediate exists only because we keep his converter minimally modified (upstream
# sync + GGTF compatibility); it is deleted on exit. Requires a key4hep env (podio).
#
# Usage: ./idea_digi_to_parquet.sh IN_digi.root OUT_canonical.parquet [--max-events N] [--stack LABEL]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
IN=$1; OUT=$2; shift 2
MAXEV=""; STACK="sw-nightlies"
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-events) MAXEV="--max-events $2"; shift 2;;
    --stack) STACK=$2; shift 2;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done
TMP=$(mktemp --suffix=.parquet 2>/dev/null || mktemp -t graphXXXX.parquet)
trap 'rm -f "$TMP" "$TMP.tmp"' EXIT
python "$HERE/../production/idea/src/process_tree.py" "$IN" "$TMP" --file-number 0 $MAXEV
python "$HERE/idea.py" "$TMP" "$OUT" --stack "$STACK"
python "$HERE/schema.py" "$OUT"
