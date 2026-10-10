import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


#load the CSV files
train_df = pd.read_csv("data/preprocessed/train.csv")
valid_df = pd.read_csv("data/preprocessed/valid.csv")
test_df = pd.read_csv("data/preprocessed/test.csv")

#different LLM names
labels = sorted(train_df["model"].unique())
label_to_idx = {}

for i in range(len(labels)):
    label_to_idx[labels[i]] = i

def tokenize(text):
    #tokenize the text into words
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def collate_fn(batch):
    #pad the sequences in the batch to the same length
    texts, lengths, ys = [], [], []
    for i, y in batch:
        texts.append(torch.tensor(i))
        lengths.append(len(i))
        ys.append(y)

    max_length = max(lengths)
    padded_texts = []

    for ids in texts:
        padded_texts.append(ids + [0] * (max_length - len(ids)))

    return torch.tensor(padded_texts), torch.tensor(lengths), torch.tensor(ys)

def build_vocab(input_type = "response", batch_size = 32):
    #turn text into tokens
    all_tokens = []
    for df in (train_df, valid_df, test_df):
        prompts = list(df[input_type])
        responses = list(df["response"])
        tokens = []
        for i in range(len(df)):
            prompt = prompts[i]
            response = responses[i]
            if input_type == "response":
                token = tokenize(response)
            elif input_type == "prompt":
                token = tokenize(prompt)
            else:
                token = tokenize(prompt)[:64] + ["<SEP>"] + tokenize(response)

            tokens.append(token[:256])
    return tokens


#mini-batch generator for training
load_train = make_loader(train_tokens, train_df, vocab, batch_size, shuffle=True)
load_valid = make_loader(valid_tokens, valid_df, vocab, batch_size, shuffle=False)
load_test = make_loader(test_tokens, test_df, vocab, batch_size, shuffle=False)
