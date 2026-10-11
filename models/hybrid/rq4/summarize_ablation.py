"""RQ4 part 2: table of the ablation results (mean ± std over 3 seeds).

Run from the repo root:
    python models/hybrid/rq4/summarize_ablation.py
Reads models/hybrid/rq4/results/ablation/<setting>/seed<k>/combined_<model>_output.json
and writes ablation_results.csv and ablation_results.md next to them.
"""

import glob
import json
import os

import pandas as pd

ROOT = "models/hybrid/rq4/results/ablation"
SETTINGS = ["baseline", "no_markdown", "truncated_50", "no_whitespace"]
MODELS = {"team_cnn": "CNN", "fusion": "Parallel"}

# one row per run
rows = []
for path in glob.glob(os.path.join(ROOT, "*", "seed*", "combined_*_output.json")):
    r = json.load(open(path))
    setting = path.replace("\\", "/").split("/")[-3]
    rows.append({"model": MODELS[r["model"]], "setting": setting, "seed": r["args"]["seed"],
                 "accuracy": r["test"]["accuracy"], "macro_f1": r["test"]["macro_f1"]})
runs = pd.DataFrame(rows)

# mean and std over the seeds
table = runs.groupby(["model", "setting"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
table.columns = ["acc_mean", "acc_std", "f1_mean", "f1_std"]
table["n_seeds"] = runs.groupby(["model", "setting"]).size()
table = table.reset_index()

# drop versus the baseline of the same model (positive = worse than baseline)
base = table[table["setting"] == "baseline"].set_index("model")
table["acc_drop"] = table["model"].map(base["acc_mean"]) - table["acc_mean"]
table["f1_drop"] = table["model"].map(base["f1_mean"]) - table["f1_mean"]

# keep the settings in a fixed order
table["setting"] = pd.Categorical(table["setting"], SETTINGS, ordered=True)
table = table.sort_values(["model", "setting"])
table.round(4).to_csv(os.path.join(ROOT, "ablation_results.csv"), index=False)

# markdown version
lines = ["| model | setting | test accuracy | test macro-F1 | acc drop | macro-F1 drop | seeds |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
for _, t in table.iterrows():
    lines.append(f"| {t['model']} | {t['setting']} | {t['acc_mean']:.3f} ± {t['acc_std']:.3f} "
                 f"| {t['f1_mean']:.3f} ± {t['f1_std']:.3f} | {t['acc_drop']:+.3f} "
                 f"| {t['f1_drop']:+.3f} | {t['n_seeds']} |")
md = "\n".join(lines) + "\n"
with open(os.path.join(ROOT, "ablation_results.md"), "w", encoding="utf-8") as f:
    f.write(md)
print(md)
