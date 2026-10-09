#!/usr/bin/env bash
# Repeat the full run_all.sh sweep over several seeds, then aggregate.
# Each seed gets its own split and initialisation; results go to results/seeds/seed<k>/.
#   ./run_seeds.sh                        # seeds 0-4, run in parallel
#   SEEDS="0 1 2" PARALLEL=0 ./run_seeds.sh --epochs 3    # extra flags go to every run
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PARALLEL="${PARALLEL:-1}"
ROOT="${ROOT:-results/seeds}"

pids=()
for seed in $SEEDS; do
  out="$ROOT/seed$seed"
  mkdir -p "$out"
  cmd=(bash run_all.sh --seed "$seed" --out_dir "$out" --ckpt_dir "checkpoints/seed$seed" "$@")
  if [[ "$PARALLEL" == "1" ]]; then
    "${cmd[@]}" > "$out/run_all.out" 2>&1 &
    pids+=($!)
    echo "seed $seed started (pid $!), log: $out/run_all.out"
  else
    "${cmd[@]}" 2>&1 | tee "$out/run_all.out"
  fi
done

status=0
for pid in "${pids[@]+"${pids[@]}"}"; do
  wait "$pid" || status=1
done
[[ $status -eq 0 ]] || { echo "at least one seed failed; see $ROOT/seed*/run_all.out" >&2; exit 1; }

"$PYTHON" aggregate_results.py --root "$ROOT"
