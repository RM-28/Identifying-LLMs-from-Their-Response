#!/usr/bin/env bash
# Run every combined model x input mode. Extra arguments are passed to each run,
# e.g.  ./run_all.sh --epochs 3 --subset 200      (quick check)
#       ./run_all.sh --cnn_ckpt models/hybrid/checkpoints/cnn.pt --rnn_ckpt models/hybrid/checkpoints/rnn.pt
# --cnn_ckpt/--rnn_ckpt only affect the ensemble runs.
set -euo pipefail
# run from the repo root so data/ and fastText/ paths work
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
OUT_DIR=models/hybrid/results

ENSEMBLE_ARGS=()
COMMON_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cnn_ckpt|--rnn_ckpt) ENSEMBLE_ARGS+=("$1" "$2"); shift 2 ;;
    --out_dir) OUT_DIR="$2"; COMMON_ARGS+=("$1" "$2"); shift 2 ;;
    *) COMMON_ARGS+=("$1"); shift ;;
  esac
done
mkdir -p "$OUT_DIR/logs"

for model in ${MODELS:-cnnlstm fusion ensemble team_cnn team_rnn}; do
  for mode in output input both; do
    extra=()
    [[ "$model" == "ensemble" ]] && extra=("${ENSEMBLE_ARGS[@]+"${ENSEMBLE_ARGS[@]}"}")
    echo "=== $model / $mode ==="
    "$PYTHON" models/hybrid/train_combined.py --model "$model" --input_mode "$mode" \
      "${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}" "${extra[@]+"${extra[@]}"}" \
      2>&1 | tee "$OUT_DIR/logs/combined_${model}_${mode}.log"
  done
done

# summary table
"$PYTHON" - "$OUT_DIR" <<'EOF'
import glob, json, os, sys
print(f"\n{'model':10s} {'input':7s} {'acc':>7s} {'macroF1':>8s}")
for path in sorted(glob.glob(os.path.join(sys.argv[1], "combined_*.json"))):
    r = json.load(open(path))
    print(f"{r['model']:10s} {r['input_mode']:7s} {r['test']['accuracy']:7.4f} {r['test']['macro_f1']:8.4f}")
EOF
