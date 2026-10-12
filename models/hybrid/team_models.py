import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(MODELS_DIR, "cnn"))
sys.path.append(os.path.join(MODELS_DIR, "rnn"))

from model.conv import CNN_NLP  # models/cnn/model/conv.py
from rnn import RNN  # models/rnn/rnn.py

MIN_LEN = 5


def pad_short_batch(ids):
    # right-pad with <pad> (id 0) if the batch is shorter than MIN_LEN
    if ids.size(1) < MIN_LEN:
        ids = F.pad(ids, (0, MIN_LEN - ids.size(1)), value=0)
    return ids


class TeamCNN(nn.Module):
    """The project CNN (CNN_NLP) as it is."""

    def __init__(self, vocab_size, num_classes, embeddings, dropout=0.5, **_):
        super().__init__()
        # clone so each model gets its own copy of the fastText weights
        self.net = CNN_NLP(pretrained_embedding=embeddings.clone(), freeze_embedding=False,
                           num_classes=num_classes, dropout=dropout)

    def forward(self, ids, lengths):
        # CNN_NLP only needs the ids
        return self.net(pad_short_batch(ids))


class TeamRNN(nn.Module):
    """The project RNN as it is."""

    def __init__(self, vocab_size, num_classes, embeddings, dropout=0.5, **_):
        super().__init__()
        self.net = RNN(input_size=vocab_size, output_size=num_classes, dropout=dropout,
                       pretrained_embeddings=embeddings.clone(), freeze_embeddings=False)

    def forward(self, ids, lengths):
        return self.net(ids, lengths)


# ---------------------------------------------------------------- combinations

class TeamCNNLSTM(nn.Module):
    """Stacked (CNN -> RNN): the CNN embedding + conv layers make a feature map,
    and the RNN reads that feature map instead of word embeddings."""

    def __init__(self, vocab_size, num_classes, embeddings, dropout=0.5, **_):
        super().__init__()
        # take the embedding and conv layers out of the CNN model
        cnn = CNN_NLP(pretrained_embedding=embeddings.clone(), freeze_embedding=False,
                      num_classes=num_classes, dropout=dropout)
        self.embedding = cnn.embedding
        self.convs = cnn.conv1d_list
        conv_dim = sum(conv.out_channels for conv in self.convs)  # 3 x 100 = 300
        # a conv with kernel k makes T - k + 1 steps, so the widest one is 4 shorter
        self.shrink = max(conv.kernel_size[0] for conv in self.convs) - 1
        self.rnn = RNN(input_size=1, output_size=num_classes, dropout=dropout,
                       pretrained_embeddings=torch.zeros(1, conv_dim))
        self.rnn.embedding = nn.Identity()

    def forward(self, ids, lengths):
        ids = pad_short_batch(ids)
        x_embed = self.embedding(ids).float()
        x_reshaped = x_embed.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.convs]
        steps = ids.size(1) - self.shrink
        x_conv_list = [x_conv[:, :, :steps] for x_conv in x_conv_list]
        features = torch.cat(x_conv_list, dim=1).permute(0, 2, 1)  # (batch, steps, 300)
        conv_lengths = (lengths - self.shrink).clamp(min=1)
        return self.rnn(features, conv_lengths)


class TeamFusion(nn.Module):
    def __init__(self, vocab_size, num_classes, embeddings, dropout=0.5, **_):
        super().__init__()
        self.cnn = TeamCNN(vocab_size, num_classes, embeddings, dropout)
        self.rnn = TeamRNN(vocab_size, num_classes, embeddings, dropout)
        # both branches use the CNN's embedding layer
        self.rnn.net.embedding = self.cnn.net.embedding

        cnn_dim = self.cnn.net.fc.in_features  # 300
        rnn_dim = self.rnn.net.fc.in_features  # 128
        # remove the CNN and RNN classifiers so forward() returns features
        self.cnn.net.fc = nn.Identity()
        self.rnn.net.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_dim + rnn_dim, num_classes)

    def forward(self, ids, lengths):
        cnn_features = self.cnn(ids, lengths)
        rnn_features = self.rnn(ids, lengths)
        features = torch.cat([cnn_features, rnn_features], dim=1)
        return self.fc(self.dropout(features))
