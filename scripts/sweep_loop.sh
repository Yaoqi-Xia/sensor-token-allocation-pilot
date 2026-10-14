#!/bin/bash
# Run the allocation sweep (124 allocations per frame) in 8 resumable pieces; finished pieces are skipped.
# Needs: XFI_DIR = your X-Fi clone; optional PYTHON (default python), PILOT_RESULTS, MMFI_ROOT.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
RES=${PILOT_RESULTS:-$HERE/results}
cd "$XFI_DIR/MMFi_HAR"
for i in 0 1 2 3 4 5 6 7; do
  out=$RES/alloc_full_p$i.npz
  [ -f "$out" ] && continue
  ${PYTHON:-python} -u "$HERE/src/eval_alloc.py" --batch_size 16 --workers 3 \
    --chunk $i/8 --out "$out" > "$RES/eval_full_p$i.log" 2>&1 || { echo "PIECE $i FAILED"; exit 1; }
  echo "PIECE $i DONE $(date +%H:%M)"
done
echo SWEEP_ALL_DONE
