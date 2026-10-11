"""The teammates' CNN and RNN, and the CNN+RNN combinations built from them.

CNN_NLP comes from models/cnn/model/conv.py (Kenny) and RNN from models/rnn/rnn.py
(Matteo). Both are imported unchanged and keep their own hyperparameters:
CNN = filters 3/4/5 x 100, dropout 0.5; RNN = unidirectional LSTM, hidden 128,
dropout 0.5. Both start from the same fastText embeddings, as in their scripts.

All models here take (ids, lengths) and return logits.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# make the teammates' folders importable (they are not packages)
MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(MODELS_DIR, "cnn"))
sys.path.append(os.path.join(MODELS_DIR, "rnn"))

from model.conv import CNN_NLP  # models/cnn/model/conv.py
from rnn import RNN  # models/rnn/rnn.py

# CNN_NLP has no padding in its convs, so the input must be at least as long
# as the widest filter (5)
MIN_LEN = 5


def pad_short_batch(ids):
    # right-pad with <pad> (id 0) if the batch is shorter than MIN_LEN
    if ids.size(1) < MIN_LEN:
        ids = F.pad(ids, (0, MIN_LEN - ids.size(1)), value=0)
    return ids


class TeamCNN(nn.Module):
    """Kenny's CNN_NLP as it is."""

    def __init__(self, vocab_size, num_classes, embeddings, dropout=0.5, **_):
        super().__init__()
        # clone so each model gets its own copy of the fastText weights
        self.net = CNN_NLP(pretrained_embedding=embeddings.clone(), freeze_embedding=False,
                           num_classes=num_classes, dropout=dropout)

    def forward(self, ids, lengths):
        # CNN_NLP only needs the ids
        return self.net(pad_short_batch(ids))


class TeamRNN(nn.Module):
    """Matteo's RNN as it is."""

    def __init__(self, vocab_size, num_classes, embeddings, dropout=0.5, **_):
        super().__init__()
        self.net = RNN(input_size=vocab_size, output_size=num_classes, dropout=dropout,
                       pretrained_embeddings=embeddings.clone(), freeze_embeddings=False)

    def forward(self, ids, lengths):
        return self.net(ids, lengths)


# ---------------------------------------------------------------- combinations

class TeamCNNLSTM(nn.Module):
    """Stacked (CNN -> RNN): Kenny's embedding + conv layers make a feature map,
    and Matteo's RNN reads that feature map instead of word embeddings."""

    def __init__(self, vocab_size, num_classes, embeddings, dropout=0.5, **_):
        super().__init__()
        # take the embedding and conv layers out of Kenny's model
        cnn = CNN_NLP(pretrained_embedding=embeddings.clone(), freeze_embedding=False,
                      num_classes=num_classes, dropout=dropout)
        self.embedding = cnn.embedding
        self.convs = cnn.conv1d_list
        conv_dim = sum(conv.out_channels for conv in self.convs)  # 3 x 100 = 300
        # a conv with kernel k makes T - k + 1 steps, so the widest one is 4 shorter
        self.shrink = max(conv.kernel_size[0] for conv in self.convs) - 1

        # Matteo's RNN. A dummy (1 x conv_dim) embedding makes his LSTM expect
        # conv_dim inputs; then the embedding is replaced by Identity, because the
        # conv features are already vectors. His forward (dropout, packing, last
        # hidden state, fc) is used unchanged.
        self.rnn = RNN(input_size=1, output_size=num_classes, dropout=dropout,
                       pretrained_embeddings=torch.zeros(1, conv_dim))
        self.rnn.embedding = nn.Identity()

    def forward(self, ids, lengths):
        ids = pad_short_batch(ids)
        # Copied from CNN_NLP.forward (models/cnn/model/conv.py), the lines before
        # max pooling. We cannot call CNN_NLP.forward because it pools over time,
        # and the LSTM needs the whole sequence.
        x_embed = self.embedding(ids).float()
        x_reshaped = x_embed.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.convs]

        # cut every feature map to the shortest length so they can be stacked
        steps = ids.size(1) - self.shrink
        x_conv_list = [x_conv[:, :, :steps] for x_conv in x_conv_list]
        features = torch.cat(x_conv_list, dim=1).permute(0, 2, 1)  # (batch, steps, 300)

        # the feature map is also shorter than the text, so shorten the lengths too
        conv_lengths = (lengths - self.shrink).clamp(min=1)
        return self.rnn(features, conv_lengths)


class TeamFusion(nn.Module):
    """Parallel: Kenny's CNN and Matteo's RNN read the same embedding layer,
    their features (300 max-pooled + 128 last hidden) are concatenated, and
    one Linear layer classifies."""

    def __init__(self, vocab_size, num_classes, embeddings, dropout=0.5, **_):
        super().__init__()
        self.cnn = TeamCNN(vocab_size, num_classes, embeddings, dropout)
        self.rnn = TeamRNN(vocab_size, num_classes, embeddings, dropout)
        # both branches use the CNN's embedding layer
        self.rnn.net.embedding = self.cnn.net.embedding

        cnn_dim = self.cnn.net.fc.in_features  # 300
        rnn_dim = self.rnn.net.fc.in_features  # 128
        # remove their classifiers so their forward() returns features
        self.cnn.net.fc = nn.Identity()
        self.rnn.net.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_dim + rnn_dim, num_classes)

    def forward(self, ids, lengths):
        cnn_features = self.cnn(ids, lengths)
        rnn_features = self.rnn(ids, lengths)
        features = torch.cat([cnn_features, rnn_features], dim=1)
        return self.fc(self.dropout(features))
