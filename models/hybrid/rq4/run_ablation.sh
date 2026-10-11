#!/usr/bin/env bash
# RQ4 ablation: retrain Kenny's CNN (team_cnn) and the Parallel model (fusion)
# on 4 versions of the text, 3 seeds each, with the unchanged train_combined.py.
# Run python models/hybrid/rq4/make_ablation_data.py first.
#   bash models/hybrid/rq4/run_ablation.sh
set -euo pipefail
# run from the repo root so data/ and fastText/ paths work
cd "$(dirname "$0")/../../.."

PYTHON="${PYTHON:-python}"
OUT=models/hybrid/rq4/results/ablation

run_seed() {
  seed=$1
  for setting in baseline no_markdown truncated_50 no_whitespace; do
    # pick the data folder and max_len for this setting
    data_dir=data/preprocessed
    max_len=256
    if [[ "$setting" == "no_markdown" || "$setting" == "no_whitespace" ]]; then
      data_dir=models/hybrid/rq4/data/$setting
    fi
    if [[ "$setting" == "truncated_50" ]]; then
      max_len=50
    fi
    for model in team_cnn fusion; do
      out_dir=$OUT/$setting/seed$seed
      mkdir -p "$out_dir/logs"
      echo "=== seed $seed / $setting / $model ==="
      "$PYTHON" models/hybrid/train_combined.py --model "$model" --input_mode output \
        --seed "$seed" --data_dir "$data_dir" --max_len "$max_len" \
        --out_dir "$out_dir" --ckpt_dir "models/hybrid/rq4/checkpoints/$setting/seed$seed" \
        > "$out_dir/logs/${model}.log" 2>&1
      grep "^TEST" "$out_dir/logs/${model}.log"
    done
  done
}

# the 3 seeds run at the same time, each one goes through all settings
for seed in 0 1 2; do
  run_seed "$seed" &
done
wait

"$PYTHON" models/hybrid/rq4/summarize_ablation.py
