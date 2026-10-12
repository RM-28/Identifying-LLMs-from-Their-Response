from rnn import RNN

import json, os, re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset


nltk.download('punkt_tab', quiet = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
INPUT_TYPE = "both"

torch.manual_seed(SEED)
np.random.seed(SEED)

train_df = pd.read_csv("data/preprocessed/train.csv")
valid_df = pd.read_csv("data/preprocessed/valid.csv")
test_df = pd.read_csv("data/preprocessed/test.csv")

labels = sorted(train_df["model"].unique())
label_to_idx = {}
for i in range(len(labels)):
    label_to_idx[labels[i]] = i

#tokenize lowercase words and punctuation

#training set
train_tokens = []
for i in range(len(train_df)):
    prompt = train_df["prompt"][i]
    response = train_df["response"][i]
    if INPUT_TYPE == "response":
        token = word_tokenize(response)
    elif INPUT_TYPE == "prompt":
        token = word_tokenize(prompt)
    else:
        token = word_tokenize(prompt)[:64] + ["<SEP>"] + word_tokenize(response)
    train_tokens.append(token[:256])

#validation set
valid_tokens = []
for i in range(len(valid_df)):
    prompt = valid_df["prompt"][i]
    response = valid_df["response"][i]
    if INPUT_TYPE == "response":
        token = word_tokenize(response)
    elif INPUT_TYPE == "prompt":
        token = word_tokenize(prompt)
    else:
        token = word_tokenize(prompt)[:64] + ["<SEP>"] + word_tokenize(response)
    valid_tokens.append(token[:256])

#test set
test_tokens = []
for i in range(len(test_df)):
    prompt = test_df["prompt"][i]
    response = test_df["response"][i]
    if INPUT_TYPE == "response":
        token = word_tokenize(response)
    elif INPUT_TYPE == "prompt":
        token = word_tokenize(prompt)
    else:
        token = word_tokenize(prompt)[:64] + ["<SEP>"] + word_tokenize(response)
    test_tokens.append(token[:256])

#build vocabulary from tokens
counts = Counter()
for t in train_tokens:
    counts.update(t)

vocab = {"<pad>": 0, "<unk>": 1}
for word, count in counts.most_common():
    if count >= 2:
        vocab[word] = len(vocab)

print(f"Vocabulary size: {len(vocab)}")

#use fastText pretrained embeddings
embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), 300))
embeddings[0] = 0
found = 0
with open("fastText/crawl-300d-2M.vec", encoding = "utf-8", errors = "ignore") as f:
    f.readline()
    for line in tqdm(f, desc="Loading fastText", total=2000000):
        parts = line.rstrip().split(" ")
        if parts[0] in vocab:
            embeddings[vocab[parts[0]]] = np.array(parts[1:], dtype = np.float32)
            found += 1

print(f"Pretrained vectors found for {found} / {len(vocab)} words")
embeddings = torch.tensor(embeddings, dtype = torch.float32)

#encode tokens to indices
train_ids = []
train_length = []
train_labels = []
for i in range(len(train_tokens)):
    ids = []
    for word in train_tokens[i]:
        ids.append(vocab.get(word, 1))

    if len(ids) == 0:
        ids = [1]

    train_length.append(len(ids))
    train_ids.append(ids + [0] * (256 - len(ids)))
    train_labels.append(label_to_idx[train_df["model"][i]])

train_data = TensorDataset(torch.tensor(train_ids), torch.tensor(train_length), torch.tensor(train_labels))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

#validation set
valid_ids = []
valid_length = []
valid_labels = []
for i in range(len(valid_tokens)):
    ids = []
    for word in valid_tokens[i]:
        ids.append(vocab.get(word, 1))

    if len(ids) == 0:
        ids = [1]

    valid_length.append(len(ids))
    valid_ids.append(ids + [0] * (256 - len(ids)))
    valid_labels.append(label_to_idx[valid_df["model"][i]])

#load validation data into DataLoader
valid_data = TensorDataset(torch.tensor(valid_ids), torch.tensor(valid_length), torch.tensor(valid_labels))
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

#test set
test_ids = []
test_length = []
test_labels = []
for i in range(len(test_tokens)):
    ids = []
    for word in test_tokens[i]:
        ids.append(vocab.get(word, 1))

    if len(ids) == 0:
        ids = [1]

    test_length.append(len(ids))
    test_ids.append(ids + [0] * (256 - len(ids)))
    test_labels.append(label_to_idx[test_df["model"][i]])

test_data = TensorDataset(torch.tensor(test_ids), torch.tensor(test_length), torch.tensor(test_labels))
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

#build the RNN model
model = RNN(input_size = len(vocab), output_size = len(labels), bidirectional = False, pretrained_embeddings = embeddings, freeze_embeddings = False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_function = nn.CrossEntropyLoss()

os.makedirs("results/metrics_rnn", exist_ok=True)
path = f"results/metrics_rnn/{INPUT_TYPE}_rnn_metrics.json"

best_val_f1 = -1
bad_epochs = 0

#train the model
for epoch in range(12):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc = f"Epoch {epoch + 1}")
    for x, lengths, y in progress:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x, lengths)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss = f"{loss.item():.4f}")

    #validation
    model.eval()
    golds, preds = [], []
    val_loss = 0
    with torch.no_grad():
        for x, lengths, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x, lengths)
            val_loss += loss_function(output, y).item()
            pred = torch.argmax(output, dim=1)
            golds.extend(y.cpu().numpy())
            preds.extend(pred.cpu().numpy())

    val_f1 = f1_score(golds, preds, average='weighted')
    print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val loss: {val_loss/len(valid_loader)}, Val F1: {val_f1:.4f}")

    #early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        bad_epochs = 0
        torch.save(model.state_dict(), f"results/metrics_rnn/{INPUT_TYPE}_best_model.pt")
    else:
        bad_epochs += 1
        if bad_epochs >= 3:
            print("Early stopping triggered.")
            break

#load the best model
model.load_state_dict(torch.load(f"results/metrics_rnn/{INPUT_TYPE}_best_model.pt", weights_only = True))
model.eval()

golds, preds = [], []
with torch.no_grad():
    for x, lengths, y in test_loader:
        out = model(x.to(device), lengths)
        preds += out.argmax(dim=1).cpu().numpy().tolist()
        golds += y.tolist()


results = {
    "accuracy": accuracy_score(golds, preds),
    "f1_score": f1_score(golds, preds, average='weighted'),
    "classification_report": classification_report(golds, preds, target_names=labels, output_dict=True),
    "confusion_matrix": confusion_matrix(golds, preds).tolist()
}

with open(path, "w") as f:
    json.dump(results, f)

print(f"\nTest accuracy: {results['accuracy']:.4f}")
print(f"F1 Score: {results['f1_score']:.4f}")
print(classification_report(golds, preds, target_names = labels))
print("Confusion Matrix:")
print(pd.DataFrame(results['confusion_matrix'], index = labels, columns = labels))