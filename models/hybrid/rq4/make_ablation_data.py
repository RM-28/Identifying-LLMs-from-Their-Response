import os
import re

import pandas as pd


def strip_markdown(text):
    # remove markdown / formatting symbols, keep the words and the newlines
    text = re.sub(r"```\w*", "", text)  # code fences (and the language name after them)
    text = text.replace("`", "")  # inline code
    text = text.replace("**", "").replace("__", "")  # bold
    text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", text)  # headers at line start
    text = re.sub(r"(?m)^[ \t]*[-*+•][ \t]+", "", text)  # bullet markers at line start
    text = re.sub(r"(?m)^[ \t]*\d+[.)][ \t]+", "", text)  # "1." / "1)" at line start
    return text


def remove_whitespace(text):
    # every run of spaces, tabs and newlines becomes one space
    return re.sub(r"\s+", " ", text).strip()


SETTINGS = {"no_markdown": strip_markdown, "no_whitespace": remove_whitespace}

for name, change in SETTINGS.items():
    out_dir = os.path.join("models/hybrid/rq4/data", name)
    os.makedirs(out_dir, exist_ok=True)
    for split in ["train", "valid", "test"]:
        df = pd.read_csv(f"data/preprocessed/{split}.csv", keep_default_na=False)
        df["response"] = df["response"].apply(change)
        df.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)
    print("wrote", out_dir)
