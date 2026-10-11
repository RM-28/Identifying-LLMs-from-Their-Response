"""Train and evaluate the CNN+RNN combination models.

Examples:
    python models/hybrid/train_combined.py --model cnnlstm --input_mode output
    python models/hybrid/train_combined.py --model ensemble --input_mode both \
        --cnn_ckpt models/hybrid/checkpoints/cnn.pt --rnn_ckpt models/hybrid/checkpoints/rnn.pt
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, Dataset

from combined_eval import bootstrap_ci, compute_metrics, plot_confusion
from combined_models import MODELS, SoftVoteEnsemble, load_member

# Kenny's fastText loader (models/cnn/load_embeddings.py)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cnn"))
from load_embeddings import load_pretrained_vectors

nltk.download("punkt_tab", quiet=True)  # needed by word_tokenize, as in models/rnn/train.py

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
# preprocess.py writes model/prompt/response; also accept the LLM_* naming
COLUMN_ALIASES = {"LLM_name": "model", "LLM_Input": "prompt", "LLM_output": "response"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["cnnlstm", "fusion", "ensemble", "team_cnn", "team_rnn"], required=True)
    p.add_argument("--input_mode", choices=["output", "input", "both"], default="output")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs)")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--emb_dim", type=int, help="unused: the team models keep their own sizes", default=128)
    p.add_argument("--num_filters", type=int, help="unused: the team models keep their own sizes", default=64)
    p.add_argument("--hidden_dim", type=int, help="unused: the team models keep their own sizes", default=128)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_vocab", type=int, default=50000)
    p.add_argument("--split", choices=["stratified", "preprocessed"], default="preprocessed",
                   help="stratified: fresh 70/15/15 split stratified by label and grouped by "
                        "prompt; preprocessed: reuse the train/valid/test CSVs as-is")
    p.add_argument("--cnn_ckpt", help="teammate CNN checkpoint for --model ensemble")
    p.add_argument("--rnn_ckpt", help="teammate RNN checkpoint for --model ensemble")
    p.add_argument("--subset", type=int, default=0, help="use only N examples per label (smoke test)")
    p.add_argument("--data_dir", default="data/preprocessed")
    p.add_argument("--fasttext", default="fastText/crawl-300d-2M.vec", help="from models/cnn/embedding_download.py")
    p.add_argument("--out_dir", default="models/hybrid/results")
    p.add_argument("--ckpt_dir", default="models/hybrid/checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data

def load_data(data_dir):
    parts = []
    for name in ("train", "valid", "test"):
        part = pd.read_csv(os.path.join(data_dir, f"{name}.csv"), keep_default_na=False)
        part["orig_split"] = name
        parts.append(part)
    df = pd.concat(parts, ignore_index=True).rename(columns=COLUMN_ALIASES)
    missing = {"model", "prompt", "response"} - set(df.columns)
    if missing:
        raise ValueError(f"{data_dir} is missing columns {missing}; found {list(df.columns)}")
    return df


def assign_split(df, how, seed):
    if how == "preprocessed":
        return df["orig_split"].replace({"valid": "val"}).to_numpy()
    # 20 folds stratified by label and grouped by prompt: 14 train / 3 val / 3 test.
    # Grouping keeps the two answers to one arena prompt in the same split.
    groups = df["prompt"].str.strip().str.lower()
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    split = np.empty(len(df), dtype=object)
    for fold, (_, idx) in enumerate(sgkf.split(df, df["model"], groups)):
        split[idx] = "train" if fold < 14 else "val" if fold < 17 else "test"
    return split


def tokenize(text):
    # same tokenizer as the teammates' CNN and RNN scripts
    return word_tokenize(text)


def build_vocab(token_lists, min_freq, max_vocab):
    counts = Counter(t for toks in token_lists for t in toks)
    vocab = {PAD: 0, UNK: 1, SEP: 2}
    for tok, c in counts.most_common():
        if c < min_freq or len(vocab) >= max_vocab:
            break
        vocab[tok] = len(vocab)
    return vocab


def select_tokens(prompt_toks, resp_toks, mode, max_len):
    if mode == "output":
        return resp_toks[:max_len]
    if mode == "input":
        return prompt_toks[:max_len]
    # both: prompt <sep> response. The response gets priority; the prompt keeps at
    # most a quarter of the budget unless the response is short.
    budget = max_len - 1
    resp = resp_toks[:budget - min(len(prompt_toks), budget // 4)]
    prompt = prompt_toks[:budget - len(resp)]
    return prompt + [SEP] + resp


class TextDataset(Dataset):
    """Each item holds one encoding per vocabulary ("view") plus the label."""

    def __init__(self, token_lists, labels, vocabs):
        self.views = [[torch.tensor([v.get(t, 1) for t in toks] or [1]) for toks in token_lists]
                      for v in vocabs]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return [view[i] for view in self.views], self.labels[i]


def collate(batch):
    seqs, labels = zip(*batch)
    views = []
    for v in range(len(seqs[0])):
        items = [s[v] for s in seqs]
        lengths = torch.tensor([len(x) for x in items])
        ids = nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=0)
        views.append((ids, lengths))
    return views, torch.stack(labels)


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def predict(forward, loader, device):
    preds, golds = [], []
    for views, y in loader:
        views = [(ids.to(device), lens.to(device)) for ids, lens in views]
        preds.append(forward(views).argmax(dim=1).cpu())
        golds.append(y)
    return torch.cat(golds).numpy(), torch.cat(preds).numpy()


def train_model(model, loaders, args, device, tag):
    """Adam + weight decay, early stopping on validation macro-F1. View 0 is the model's input."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    forward = lambda views: model(*views[0])
    best_f1, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for views, y in loaders["train"]:
            ids, lens = views[0]
            ids, lens, y = ids.to(device), lens.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(ids, lens), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
            n += len(y)

        model.eval()
        y_true, y_pred = predict(forward, loaders["val"], device)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        val_acc = float((y_true == y_pred).mean())
        history.append({"epoch": epoch, "train_loss": total / n, "val_macro_f1": val_f1, "val_acc": val_acc})
        print(f"[{tag}] epoch {epoch:2d}  loss {total / n:.4f}  val_acc {val_acc:.4f}  "
              f"val_macroF1 {val_f1:.4f}  ({time.time() - t0:.1f}s)", flush=True)

        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{tag}] early stop: no val macro-F1 gain for {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_macro_f1": best_f1, "best_epoch": best_epoch, "history": history}


def model_kwargs(args, vocab_size, num_classes):
    return {"vocab_size": vocab_size, "num_classes": num_classes, "emb_dim": args.emb_dim,
            "num_filters": args.num_filters, "hidden_dim": args.hidden_dim, "dropout": args.dropout}


def save_checkpoint(path, name, model, kwargs, vocab, labels, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_class": f"{MODELS[name].__module__}:{MODELS[name].__name__}",
                "model_kwargs": kwargs, "state_dict": model.state_dict(),
                "vocab": vocab, "labels": labels, "args": vars(args)}, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    # deterministic cuDNN kernels: the same seed reproduces the same numbers
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    run_name = f"combined_{args.model}_{args.input_mode}"

    df = load_data(args.data_dir)
    if args.subset:
        df = (df.sample(frac=1, random_state=args.seed)
                .groupby("model").head(args.subset).reset_index(drop=True))
    labels = sorted(df["model"].unique())
    label_id = {name: i for i, name in enumerate(labels)}
    df["y"] = df["model"].map(label_id)
    df["split"] = assign_split(df, args.split, args.seed)
    df["tokens"] = [select_tokens(tokenize(p), tokenize(r), args.input_mode, args.max_len)
                    for p, r in zip(df["prompt"], df["response"])]

    sizes = df.groupby(["split", "model"]).size().unstack(fill_value=0)
    print(f"{run_name}  device={device}\n{sizes}\n")

    train_df = df[df["split"] == "train"]
    vocab = build_vocab(train_df["tokens"], args.min_freq, args.max_vocab)
    print(f"vocab size: {len(vocab)}")

    # fastText vectors for our vocab; words not in fastText get random vectors
    embeddings = load_pretrained_vectors(vocab, args.fasttext)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)

    # ensemble members: teammate checkpoints if given, otherwise TeamCNN/TeamRNN trained here
    members, member_vocabs, member_info, train_logs = [], [], {}, {}
    if args.model == "ensemble":
        for role, ckpt, standin in (("cnn", args.cnn_ckpt, "team_cnn"), ("rnn", args.rnn_ckpt, "team_rnn")):
            if ckpt:
                model, own_vocab = load_member(ckpt, map_location=device)
                members.append(model.to(device))
                member_vocabs.append(own_vocab or vocab)
                member_info[role] = {"source": ckpt, "own_vocab": own_vocab is not None}
            else:
                members.append(standin)  # placeholder, trained below
                member_vocabs.append(vocab)
                member_info[role] = {"source": f"{standin} trained by train_combined.py"}
        views = member_vocabs
    else:
        views = [vocab]

    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        ds = TextDataset(part["tokens"].tolist(), part["y"].tolist(), views)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"),
                                    collate_fn=collate,
                                    generator=torch.Generator().manual_seed(args.seed))

    t_start = time.time()
    if args.model == "ensemble":
        for i, (role, m) in enumerate(zip(("cnn", "rnn"), list(members))):
            if isinstance(m, str):  # train the member on view i (shares our vocab)
                kwargs = model_kwargs(args, len(vocab), len(labels))
                model = MODELS[m](embeddings=embeddings, **kwargs)
                member_loaders = {k: _ViewLoader(v, i) for k, v in loaders.items()}
                train_logs[role] = train_model(model, member_loaders, args, device, f"{run_name}/{role}")
                save_checkpoint(os.path.join(args.ckpt_dir, f"standin_{m}_{args.input_mode}.pt"),
                                m, model, kwargs, vocab, labels, args)
                members[i] = model
        net = SoftVoteEnsemble(members).to(device).eval()
        forward = net
        # per-member test scores, for comparison with the ensemble
        for i, role in enumerate(("cnn", "rnn")):
            y_true, y_pred = predict(lambda v, i=i: net.member_logits(net.members[i], *v[i]),
                                     loaders["test"], device)
            member_info[role]["test"] = {k: v for k, v in compute_metrics(y_true, y_pred, labels).items()
                                         if k in ("accuracy", "macro_f1")}
    else:
        kwargs = model_kwargs(args, len(vocab), len(labels))
        net = MODELS[args.model](embeddings=embeddings, **kwargs)
        train_logs[args.model] = train_model(net, loaders, args, device, run_name)
        save_checkpoint(os.path.join(args.ckpt_dir, f"{run_name}.pt"),
                        args.model, net, kwargs, vocab, labels, args)
        forward = lambda v: net(*v[0])

    y_true, y_pred = predict(forward, loaders["test"], device)
    metrics = compute_metrics(y_true, y_pred, labels)
    metrics["ci95"] = bootstrap_ci(y_true, y_pred, seed=args.seed)
    y_vtrue, y_vpred = predict(forward, loaders["val"], device)

    os.makedirs(args.out_dir, exist_ok=True)
    png = os.path.join(args.out_dir, f"{run_name}_confusion.png")
    plot_confusion(metrics["confusion_matrix"], labels, png,
                   f"{args.model} · input mode: {args.input_mode} · test acc {metrics['accuracy']:.3f}, "
                   f"macro-F1 {metrics['macro_f1']:.3f}")

    result = {
        "model": args.model, "input_mode": args.input_mode, "args": vars(args),
        "labels": labels, "vocab_size": len(vocab),
        "split_sizes": {s: {k: int(v) for k, v in row.items()} for s, row in sizes.iterrows()},
        "val_macro_f1": float(f1_score(y_vtrue, y_vpred, average="macro")),
        "test": metrics, "training": train_logs, "train_time_sec": round(time.time() - t_start, 1),
        "confusion_png": png,
        # per-row test predictions, so models run with the same seed (same test
        # set) can be compared pairwise by aggregate_results.py
        "test_predictions": {"row_id": df.index[df["split"] == "test"].tolist(),
                             "y_true": y_true.tolist(), "y_pred": y_pred.tolist()},
    }
    if member_info:
        result["members"] = member_info
    out = os.path.join(args.out_dir, f"{run_name}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    ci = metrics["ci95"]
    print(f"\nTEST  acc {metrics['accuracy']:.4f} [{ci['accuracy'][0]:.3f}, {ci['accuracy'][1]:.3f}]  "
          f"macro-F1 {metrics['macro_f1']:.4f} [{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}]")
    for name, m in metrics["per_class"].items():
        print(f"  {name:15s} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  n={m['support']}")
    print(f"saved {out} and {png}")


class _ViewLoader:
    """Wraps a loader so view i becomes view 0 (lets train_model stay single-view)."""

    def __init__(self, loader, i):
        self.loader, self.i = loader, i

    def __iter__(self):
        for views, y in self.loader:
            yield [views[self.i]], y


if __name__ == "__main__":
    main()
