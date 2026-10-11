"""The teammates' CNN and RNN, and the CNN+RNN combinations built from them.

The architectures are imported unchanged from models/cnn/model/conv.py (Kenny)
and models/rnn/rnn.py (Matteo); this file only gives them the (ids, lengths)
interface and constructor kwargs the shared pipeline uses, so every model is
trained and tested on the same splits, tokens and training loop. TeamCNNLSTM
(stacked) and TeamFusion (parallel) combine their parts; the soft-vote
ensemble in combined_models.py averages TeamCNN and TeamRNN.
Their own hyperparameter defaults are kept; the shared emb_dim/num_filters/
hidden_dim flags are ignored.
"""

import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(rel_path, name):
    # load by path: the teammates' folders are not packages
    spec = importlib.util.spec_from_file_location(f"_team_{name}", os.path.join(_ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


CNN_NLP = _load(os.path.join("cnn", "model", "conv.py"), "CNN_NLP")
RNN = _load(os.path.join("rnn", "rnn.py"), "RNN")


class TeamCNN(nn.Module):
    """Kenny's CNN_NLP: random-init 300-d embeddings, filters 3/4/5 x 100, max pool."""

    MIN_LEN = 5  # widest filter; shorter batches are right-padded

    def __init__(self, vocab_size, num_classes, dropout=0.5, **_):
        super().__init__()
        self.net = CNN_NLP(vocab_size=vocab_size, num_classes=num_classes, dropout=dropout)

    def forward(self, ids, lengths):
        if ids.size(1) < self.MIN_LEN:
            ids = F.pad(ids, (0, self.MIN_LEN - ids.size(1)), value=0)
        return self.net(ids)


class TeamRNN(nn.Module):
    """Matteo's RNN: 128-d embedding, unidirectional LSTM, final hidden state."""

    def __init__(self, vocab_size, num_classes, dropout=0.5, **_):
        super().__init__()
        self.net = RNN(input_size=vocab_size, output_size=num_classes, dropout=dropout)

    def forward(self, ids, lengths):
        return self.net(ids, lengths)


# ---------------------------------------------------------------- combinations

class TeamCNNLSTM(nn.Module):
    """Stacked: Kenny's embedding + convs -> Matteo-style LSTM -> final hidden state -> Linear.

    Kenny's convs are unpadded, so kernel k gives T-k+1 steps; every map is trimmed
    to the shortest (T-4) before concatenating, and the LSTM is packed with lengths-4.
    """

    MIN_LEN = TeamCNN.MIN_LEN

    def __init__(self, vocab_size, num_classes, dropout=0.5, hidden_size=128, **_):
        super().__init__()
        cnn = CNN_NLP(vocab_size=vocab_size, num_classes=num_classes, dropout=dropout)
        self.embedding, self.convs = cnn.embedding, cnn.conv1d_list
        self.shrink = max(c.kernel_size[0] for c in self.convs) - 1
        conv_dim = sum(c.out_channels for c in self.convs)
        # same settings as Matteo's RNN.lstm; input is the conv features, not embeddings
        self.lstm = nn.LSTM(conv_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, ids, lengths):
        if ids.size(1) < self.MIN_LEN:
            ids = F.pad(ids, (0, self.MIN_LEN - ids.size(1)), value=0)
        x = self.embedding(ids).float().permute(0, 2, 1)
        T = ids.size(1) - self.shrink
        x = torch.cat([F.relu(conv(x))[:, :, :T] for conv in self.convs], dim=1).permute(0, 2, 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.dropout(x), (lengths - self.shrink).clamp(min=1).cpu(),
            batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.fc(self.dropout(hidden[-1]))


class TeamFusion(nn.Module):
    """Parallel: Kenny's CNN and Matteo's RNN, each with its own embedding, heads removed;
    their features (300 max-pooled + 128 final hidden) are concatenated -> Linear."""

    def __init__(self, vocab_size, num_classes, dropout=0.5, **_):
        super().__init__()
        self.cnn = TeamCNN(vocab_size, num_classes, dropout)
        self.rnn = TeamRNN(vocab_size, num_classes, dropout)
        cnn_dim, rnn_dim = self.cnn.net.fc.in_features, self.rnn.net.fc.in_features
        self.cnn.net.fc = nn.Identity()  # their forward() now returns features
        self.rnn.net.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_dim + rnn_dim, num_classes)

    def forward(self, ids, lengths):
        feats = torch.cat([self.cnn(ids, lengths), self.rnn(ids, lengths)], dim=1)
        return self.fc(self.dropout(feats))
