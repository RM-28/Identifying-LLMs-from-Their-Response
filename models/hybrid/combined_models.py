import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from team_models import TeamCNN, TeamCNNLSTM, TeamFusion, TeamRNN


class SoftVoteEnsemble(nn.Module):
    def __init__(self, members, weights=None):
        super().__init__()
        self.members = nn.ModuleList(members)
        w = torch.ones(len(members)) if weights is None else torch.tensor(weights, dtype=torch.float)
        self.register_buffer("weights", w / w.sum())

    @staticmethod
    def member_logits(member, ids, lengths):
        # some models only accept ids; fall back to that signature
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
