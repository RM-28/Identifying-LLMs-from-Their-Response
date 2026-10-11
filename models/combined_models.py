"""CNN + RNN combinations built from the teammates' CNN (Kenny) and RNN (Matteo).

All models take a padded batch of token ids (B, T) plus the true lengths (B,)
and return unnormalised logits (B, num_classes). Index 0 is the padding token.
The stacked and parallel models live in models/team_models.py; this file holds
the soft-vote ensemble, the checkpoint loader and the model registry.
"""

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.team_models import TeamCNN, TeamCNNLSTM, TeamFusion, TeamRNN


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


MODELS = {"cnnlstm": TeamCNNLSTM, "fusion": TeamFusion,
          "team_cnn": TeamCNN, "team_rnn": TeamRNN}
