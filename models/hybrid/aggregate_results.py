"""Aggregate multi-seed runs of train_combined.py.

Reads <root>/seed*/combined_<model>_<mode>.json (as written by run_seeds.sh) and reports:
  * mean ± std and a t-based 95% CI across seeds for test accuracy and macro-F1
  * paired comparisons between configurations. Runs with the same seed share
    the same test rows, so each seed gives one paired difference; across seeds
    we report the mean difference, wins, and a paired t-test. Each seed also gets
    an exact McNemar test on per-row correctness.
Writes <root>/summary.json and <root>/summary.md.

    python models/hybrid/aggregate_results.py --root models/hybrid/results/seeds
"""

import argparse
import glob
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

MODELS = ("cnnlstm", "fusion", "ensemble", "team_cnn", "team_rnn")
MODES = ("output", "both", "input")


def load_runs(root):
    runs = defaultdict(dict)  # (model, mode) -> {seed: result}
    for path in glob.glob(os.path.join(root, "seed*", "combined_*.json")):
        r = json.load(open(path))
        seed = int(re.search(r"seed(\d+)", path).group(1))
        runs[(r["model"], r["input_mode"])][seed] = r
    return runs


def summarize(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half], "n": n,
            "values": [float(x) for x in v]}


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar p-value from the discordant pairs."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "p": p}


def compare(runs, a, b):
    seeds = sorted(set(runs[a]) & set(runs[b]))
    diffs, wins, per_seed = [], 0, []
    for s in seeds:
        ra, rb = runs[a][s], runs[b][s]
        pa, pb = ra["test_predictions"], rb["test_predictions"]
        if pa["row_id"] != pb["row_id"]:
            raise ValueError(f"seed {s}: {a} and {b} were not evaluated on the same test rows")
        y = np.array(pa["y_true"])
        ca, cb = np.array(pa["y_pred"]) == y, np.array(pb["y_pred"]) == y
        d = ra["test"]["macro_f1"] - rb["test"]["macro_f1"]
        diffs.append(d)
        wins += d > 1e-9  # exact ties are not wins
        per_seed.append({"seed": s, "macro_f1_diff": d, "mcnemar": mcnemar_exact(ca, cb)})
    out = {"a": "/".join(a), "b": "/".join(b), "n_seeds": len(seeds), "wins_a": int(wins),
           "macro_f1_diff": summarize(diffs) if seeds else None, "per_seed": per_seed}
    if len(seeds) > 1:
        fa = [runs[a][s]["test"]["macro_f1"] for s in seeds]
        fb = [runs[b][s]["test"]["macro_f1"] for s in seeds]
        out["paired_t_p"] = float(stats.ttest_rel(fa, fb).pvalue)
    return out


def fmt(s):
    return f"{s['mean']:.3f} ± {s['std']:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="models/hybrid/results/seeds")
    args = p.parse_args()

    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"no runs found under {args.root}/seed*/")

    table = {}
    for key, by_seed in sorted(runs.items()):
        rs = [by_seed[s] for s in sorted(by_seed)]
        entry = {"seeds": sorted(by_seed),
                 "accuracy": summarize([r["test"]["accuracy"] for r in rs]),
                 "macro_f1": summarize([r["test"]["macro_f1"] for r in rs]),
                 "per_class_f1": {lab: summarize([r["test"]["per_class"][lab]["f1"] for r in rs])
                                  for lab in rs[0]["labels"]}}
        if "members" in rs[0]:
            entry["members_macro_f1"] = {role: summarize([r["members"][role]["test"]["macro_f1"] for r in rs])
                                         for role in rs[0]["members"]}
        table["/".join(key)] = entry

    comparisons = []
    for mode in MODES:  # models against each other, same input
        for a, b in itertools.combinations(MODELS, 2):
            if (a, mode) in runs and (b, mode) in runs:
                comparisons.append(compare(runs, (a, mode), (b, mode)))
    for model in MODELS:  # input modes against each other, same model
        for a, b in (("output", "both"), ("output", "input"), ("both", "input")):
            if (model, a) in runs and (model, b) in runs:
                comparisons.append(compare(runs, (model, a), (model, b)))

    summary = {"root": args.root, "results": table, "comparisons": comparisons}
    with open(os.path.join(args.root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    seeds = sorted({s for by_seed in runs.values() for s in by_seed})
    lines = [f"# All models: {len(seeds)} seeds ({', '.join(map(str, seeds))})", "",
             "Test scores as mean ± std across seeds; [95% CI of the mean, t-distribution].", "",
             "| model | input | accuracy | macro-F1 | macro-F1 95% CI |", "| --- | --- | --- | --- | --- |"]
    for mode in MODES:
        for model in MODELS:
            e = table.get(f"{model}/{mode}")
            if e:
                lo, hi = e["macro_f1"]["ci95"]
                lines.append(f"| {model} | {mode} | {fmt(e['accuracy'])} | {fmt(e['macro_f1'])} | [{lo:.3f}, {hi:.3f}] |")
    members = {k: v["members_macro_f1"] for k, v in table.items() if "members_macro_f1" in v}
    if members:
        lines += ["", "Ensemble members on their own (macro-F1):", "",
                  "| ensemble run | " + " | ".join(next(iter(members.values()))) + " |",
                  "| --- |" + " --- |" * len(next(iter(members.values())))]
        for k, m in members.items():
            lines.append(f"| {k} | " + " | ".join(fmt(s) for s in m.values()) + " |")
    lines += ["", "## Paired comparisons (A − B, macro-F1)", "",
              "Same seed means same test rows. *Wins* = seeds where A beat B. *McNemar* = seeds where "
              "per-row correctness differed at p < 0.05.", "",
              "| A | B | mean diff ± std | wins | paired t p | McNemar p<0.05 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for c in comparisons:
        sig = sum(ps["mcnemar"]["p"] < 0.05 for ps in c["per_seed"])
        tp = f"{c['paired_t_p']:.3g}" if np.isfinite(c.get("paired_t_p", np.nan)) else "–"
        lines.append(f"| {c['a']} | {c['b']} | {c['macro_f1_diff']['mean']:+.3f} ± {c['macro_f1_diff']['std']:.3f} "
                     f"| {c['wins_a']}/{c['n_seeds']} | {tp} | {sig}/{c['n_seeds']} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(args.root, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252 (no ± or −)
    print(md)


if __name__ == "__main__":
    main()
