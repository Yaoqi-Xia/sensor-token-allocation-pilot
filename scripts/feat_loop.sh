#!/bin/bash
# Extract encoder features + fusion-attention probe in 8 resumable pieces (same split as the sweep).
# Needs: XFI_DIR = your X-Fi clone; optional PYTHON (default python), PILOT_RESULTS, MMFI_ROOT.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
RES=${PILOT_RESULTS:-$HERE/results}
cd "$XFI_DIR/MMFi_HAR"
for i in 0 1 2 3 4 5 6 7; do
  out=$RES/feats_full_p$i.npz
  [ -f "$out" ] && continue
  ${PYTHON:-python} -u "$HERE/src/extract_feats.py" --chunk $i/8 --out "$out" \
    > "$RES/feats_p$i.log" 2>&1 || { echo "PIECE $i FAILED"; exit 1; }
  echo "PIECE $i DONE $(date +%H:%M)"
done
echo FEATS_ALL_DONE
