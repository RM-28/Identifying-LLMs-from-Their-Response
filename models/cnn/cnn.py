'''
grab the dataset (ig just the train set for now?)

tokenize

load a pretrained model



RQ1: make a model/object that takes a string/list of strings 
and outputs predictions of which model generated it

RQ2: need an option to classify based off just prompt, just response, or both

RQ3: tasks, it asks whether a model trained on classifying responses on math can generalize to writing, etc 
'''

import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from models.cnn.tokenize import tokenize, encode
from models.cnn.load_embeddings import load_pretrained_vectors
from models.cnn.dataloader import data_loader


# 1. Load your preprocessed CSVs
train_df = pd.read_csv("data/preprocessed/train.csv")
valid_df = pd.read_csv("data/preprocessed/valid.csv") # note i think it includes index on left
test_df = pd.read_csv("data/preprocessed/test.csv")

# 2. Map string labels (model names) to integer IDs for PyTorch
MODEL_NAMES = ["gpt-3.5-turbo", "claude-v1", "vicuna-13b", "palm-2"]
label_mapping = {name: i for i, name in enumerate(MODEL_NAMES)}

train_labels = train_df["model"].map(label_mapping).values # looks like: [1, 2, 0, 3...], numpy array
val_labels = valid_df["model"].map(label_mapping).values
test_labels = test_df["model"].map(label_mapping).values

# 3. Extract the text data into lists
train_texts = train_df["response"].tolist()
val_texts = valid_df["response"].tolist()
test_texts = test_df["response"].tolist()

# 4. Tokenize and build vocabulary ONLY on the training data
print("Tokenizing training data...")
train_tokenized, word2idx, max_len = tokenize(train_texts)

# Tokenize validation and test sets without updating the vocabulary
val_tokenized = [word_tokenize(sent) for sent in val_texts]
test_tokenized = [word_tokenize(sent) for sent in test_texts]

# 5. Encode the tokens into numpy arrays of IDs
print("Encoding text to input IDs...")
train_inputs = encode(train_tokenized, word2idx, max_len)
val_inputs = encode(val_tokenized, word2idx, max_len)
test_inputs = encode(test_tokenized, word2idx, max_len)

# 6. Load pretrained vectors (using the word2idx built from train)
embeddings = load_pretrained_vectors(word2idx, "fastText/crawl-300d-2M.vec")
embeddings = torch.tensor(embeddings)

# 7. Create PyTorch DataLoaders using Chris Tran's function
train_dataloader, val_dataloader = data_loader(
    train_inputs, # encoded 
    val_inputs, # encoded 
    train_labels, # classes encoded
    val_labels, # classes encoded
    batch_size=50
)