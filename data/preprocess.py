import os

import numpy as np
import pandas as pd
from datasets import load_dataset

#model names
MODEL_NAMES = ["gpt-3.5-turbo", "claude-v1", "vicuna-13b", "palm-2"]

#load dataset for english
df = load_dataset("lmsys/chatbot_arena_conversations", split="train").to_pandas()
df = df.query("language == 'English' and turn == 1")

#convert the dataset to a long format
rows = []
for _, r in df.iterrows():
    for side in ("a", "b"):
        rows.append(
            {
                "model": r[f"model_{side}"],
                "prompt": r[f"conversation_{side}"][0]["content"].strip(),
                "response": r[f"conversation_{side}"][1]["content"].strip(),
            }
        )

df = pd.DataFrame(rows)

#clean the data to drop empty and duplicate rows
df = df[df["model"].isin(MODEL_NAMES)]
df = df.query("prompt != '' and response != ''")
df = df.drop_duplicates()

#downsize the dataset to smallest model size
n = df["model"].value_counts().min()
df = df.groupby("model").sample(n = n, random_state=42)

#split dataset, into 70/15/15 train/valid/test splits
prompt_id = df["prompt"].str.strip().str.lower().str.join("")
ids = prompt_id.unique()
np.random.seed(42)
np.random.shuffle(ids)

n_train = int(len(ids) * 0.70)
n_valid = int(len(ids) * 0.15)

split = {}
for i, k in enumerate(ids):
    if i < n_train:
        split[k] = "train"
    elif i < n_train + n_valid:
        split[k] = "valid"
    else:
        split[k] = "test"

df["split"] = prompt_id.map(split)

#save the dataset to csv files
os.makedirs("data/preprocessed", exist_ok=True)
for name in ("train", "valid", "test"):
    part = df[df["split"] == name].drop(columns=["split"])
    part.to_csv(f"data/preprocessed/{name}.csv", index=False)

#print summary
print(pd.crosstab(df["model"], df["split"]))