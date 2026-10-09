"""CNN + RNN combination models for LLM attribution.

All models take a padded batch of token ids (B, T) plus the true lengths (B,)
and return unnormalised logits (B, num_classes). Index 0 is the padding token.
"""

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

KERNEL_SIZES = (2, 3, 4, 5)


def length_mask(lengths, max_len):
    """(B, T) bool mask, True on real tokens."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class ConvBlock(nn.Module):
    """Parallel Conv1d + ReLU branches whose outputs keep the input length."""

    def __init__(self, emb_dim, num_filters, kernel_sizes=KERNEL_SIZES):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(emb_dim, num_filters, k, padding=k // 2) for k in kernel_sizes
        )
        self.out_dim = num_filters * len(kernel_sizes)

    def forward(self, emb):
        # emb: (B, T, E) -> (B, T, out_dim). Even kernels produce T+1 outputs,
        # so every branch is trimmed back to T before concatenating.
        x = emb.transpose(1, 2)
        T = x.size(2)
        feats = [F.relu(conv(x))[:, :, :T] for conv in self.convs]
        return torch.cat(feats, dim=1).transpose(1, 2)


def run_bilstm(lstm, x, lengths):
    """Packed BiLSTM pass. Returns (outputs (B, T, 2H), final hidden (B, 2H))."""
    packed = pack_padded_sequence(
        x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
    )
    out, (h_n, _) = lstm(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
    # last layer's forward and backward final states
    final = torch.cat([h_n[-2], h_n[-1]], dim=1)
    return out, final


def masked_max_mean(x, mask):
    """Max and mean over time ignoring padding. x: (B, T, D) -> (B, 2D)."""
    m = mask.unsqueeze(-1)
    x_max = x.masked_fill(~m, float("-inf")).max(dim=1).values
    x_mean = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return torch.cat([x_max, x_mean], dim=1)


class CNNLSTM(nn.Module):
    """Embedding -> parallel Conv1d (k=2,3,4,5) -> concat -> BiLSTM -> max+mean pool -> Linear."""

    def __init__(self, vocab_size, num_classes, emb_dim=128, num_filters=64,
                 hidden_dim=128, lstm_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = ConvBlock(emb_dim, num_filters)
        self.lstm = nn.LSTM(self.conv.out_dim, hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden_dim, num_classes)

    def forward(self, ids, lengths):
        mask = length_mask(lengths, ids.size(1))
        x = self.dropout(self.embedding(ids))
        x = self.conv(x)
        out, _ = run_bilstm(self.lstm, x, lengths)
        pooled = masked_max_mean(out, mask)
        return self.fc(self.dropout(pooled))


class CNNBranch(nn.Module):
    """Parallel convs + global max pool over time -> (B, out_dim)."""

    def __init__(self, emb_dim, num_filters):
        super().__init__()
        self.conv = ConvBlock(emb_dim, num_filters)
        self.out_dim = self.conv.out_dim

    def forward(self, emb, mask):
        feats = self.conv(emb)
        return feats.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values


class ParallelFusion(nn.Module):
    """Shared embedding feeding a CNN branch and a BiLSTM branch side by side; concat -> Linear."""

    def __init__(self, vocab_size, num_classes, emb_dim=128, num_filters=64,
                 hidden_dim=128, lstm_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cnn = CNNBranch(emb_dim, num_filters)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.cnn.out_dim + 2 * hidden_dim, num_classes)

    def forward(self, ids, lengths):
        mask = length_mask(lengths, ids.size(1))
        emb = self.dropout(self.embedding(ids))
        cnn_feat = self.cnn(emb, mask)
        _, lstm_feat = run_bilstm(self.lstm, emb, lengths)
        return self.fc(self.dropout(torch.cat([cnn_feat, lstm_feat], dim=1)))


# Stand-alone members, used by the ensemble only when the teammates' CNN/RNN
# checkpoints are not available.

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=128, num_filters=64, dropout=0.5, **_):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cnn = CNNBranch(emb_dim, num_filters)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.cnn.out_dim, num_classes)

    def forward(self, ids, lengths):
        mask = length_mask(lengths, ids.size(1))
        emb = self.dropout(self.embedding(ids))
        return self.fc(self.dropout(self.cnn(emb, mask)))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=128, hidden_dim=128,
                 lstm_layers=1, dropout=0.5, **_):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, ids, lengths):
        emb = self.dropout(self.embedding(ids))
        _, final = run_bilstm(self.lstm, emb, lengths)
        return self.fc(self.dropout(final))


class SoftVoteEnsemble(nn.Module):
    """Averages the softmax probabilities of several trained classifiers.

    Members may use different vocabularies, so forward() takes one
    (ids, lengths) pair per member. It returns log(mean probability), which
    keeps the "returns logits" contract: argmax and softmax are unchanged.
    """

    def __init__(self, members, weights=None):
        super().__init__()
        self.members = nn.ModuleList(members)
        w = torch.ones(len(members)) if weights is None else torch.tensor(weights, dtype=torch.float)
        self.register_buffer("weights", w / w.sum())

    @staticmethod
    def member_logits(member, ids, lengths):
        # Teammates' models may only accept ids; fall back to that signature.
        try:
            return member(ids, lengths)
        except TypeError:
            return member(ids)

    def member_probs(self, inputs):
        return [F.softmax(self.member_logits(m, ids, lens), dim=-1)
                for m, (ids, lens) in zip(self.members, inputs)]

    def forward(self, inputs):
        probs = torch.stack(self.member_probs(inputs), dim=0)
        avg = (probs * self.weights.view(-1, 1, 1)).sum(dim=0)
        return torch.log(avg.clamp(min=1e-12))


def load_member(path, map_location="cpu"):
    """Load a teammate checkpoint as (model, vocab_or_None).

    Accepted formats (torch.save):
      * a whole nn.Module
      * {"model": nn.Module, "vocab": {token: id}}
      * {"model_class": "models.cnn:CNN", "model_kwargs": {...},
         "state_dict": {...}, "vocab": {token: id}}
    "vocab" is optional; without it the member is fed the combined pipeline's vocab.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, nn.Module):
        return ckpt.eval(), None
    vocab = ckpt.get("vocab")
    if isinstance(ckpt.get("model"), nn.Module):
        return ckpt["model"].eval(), vocab
    if "model_class" in ckpt:
        module_name, cls_name = ckpt["model_class"].split(":")
        cls = getattr(importlib.import_module(module_name), cls_name)
        model = cls(**ckpt.get("model_kwargs", {}))
        model.load_state_dict(ckpt["state_dict"])
        return model.eval(), vocab
    raise ValueError(f"Unrecognised checkpoint format in {path}: keys={list(ckpt)}")


MODELS = {"cnnlstm": CNNLSTM, "fusion": ParallelFusion,
          "cnn": CNNClassifier, "lstm": LSTMClassifier}
