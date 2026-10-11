"""RQ4 part 1: per-LLM statistics of the responses in the training data.

Run from the repo root:
    python models/hybrid/rq4/stats.py
Writes models/hybrid/rq4/results/llm_stats.csv and three plots.
"""

import os

import matplotlib

matplotlib.use("Agg")  # save plots to files, no window
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab", quiet=True)

OUT_DIR = "models/hybrid/rq4/results"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("data/preprocessed/train.csv", keep_default_na=False)
text = df["response"]

# same tokenizer as the training pipeline (train_combined.py)
tokens = text.apply(word_tokenize)

# length and lexical diversity, one value per response
df["chars"] = text.str.len()
df["tokens"] = tokens.apply(len)
# type-token ratio = unique tokens / all tokens (lowercased)
df["ttr"] = tokens.apply(lambda t: len(set(w.lower() for w in t)) / max(len(t), 1))

# formatting counts per response (regex counts; (?m) makes ^ match every line start)
df["bold"] = text.str.count(r"\*\*[^*\n]+\*\*")
df["headers"] = text.str.count(r"(?m)^\s*#{1,6}\s")
df["bullets"] = text.str.count(r"(?m)^\s*[-*+•]\s")
df["numbered"] = text.str.count(r"(?m)^\s*\d+[.)]\s")
df["newlines"] = text.str.count("\n")
df["code_blocks"] = text.str.count("```") // 2  # a block has an opening and a closing fence

# phrases, case-insensitive
df["as_an_ai"] = text.str.lower().str.count("as an ai")
df["i_apologize"] = text.str.lower().str.count("i apologize")
df["certainly"] = text.str.lower().str.count("certainly")

features = ["chars", "tokens", "ttr", "bold", "headers", "bullets", "numbered",
            "newlines", "code_blocks", "as_an_ai", "i_apologize", "certainly"]

# mean per response for each LLM
table = df.groupby("model")[features].mean()

# vocabulary size = number of different tokens the LLM uses in all its responses
table["vocab_size"] = [len(set(w.lower() for t in tokens[df["model"] == m] for w in t))
                       for m in table.index]
table["n_responses"] = df["model"].value_counts()

# share of responses that use each format / phrase at least once
for f in ["bold", "headers", "bullets", "numbered", "code_blocks",
          "as_an_ai", "i_apologize", "certainly"]:
    table["pct_with_" + f] = df.groupby("model")[f].apply(lambda x: (x > 0).mean() * 100)

table = table.round(3)
table.to_csv(os.path.join(OUT_DIR, "llm_stats.csv"))
print(table.T)

models = list(table.index)

# plot 1: response length in tokens per LLM
plt.figure(figsize=(7, 4))
plt.boxplot([df.loc[df["model"] == m, "tokens"] for m in models], showfliers=False)
plt.xticks(range(1, len(models) + 1), models)
plt.ylabel("tokens per response")
plt.title("Response length (training data, outliers hidden)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "length_tokens.png"), dpi=150)
plt.close()

# plot 2: type-token ratio per LLM
plt.figure(figsize=(7, 4))
plt.boxplot([df.loc[df["model"] == m, "ttr"] for m in models], showfliers=False)
plt.xticks(range(1, len(models) + 1), models)
plt.ylabel("type-token ratio")
plt.title("Lexical diversity per response (training data)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ttr.png"), dpi=150)
plt.close()

# plot 3: % of responses that use each format / phrase
pct_cols = ["pct_with_bold", "pct_with_headers", "pct_with_bullets", "pct_with_numbered",
            "pct_with_code_blocks", "pct_with_as_an_ai", "pct_with_i_apologize",
            "pct_with_certainly"]
ax = table[pct_cols].T.plot(kind="bar", figsize=(9, 4.5))
ax.set_xticklabels([c.replace("pct_with_", "") for c in pct_cols], rotation=30, ha="right")
ax.set_ylabel("% of responses")
ax.set_title("Formatting and phrases (training data)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "formatting.png"), dpi=150)
plt.close()

print("saved", OUT_DIR)
