"""Metrics and confusion-matrix plot shared by the combined-model experiments."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

# single-hue sequential ramp (light -> dark blue); lightest step = "near zero"
CM_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
INK = "#1f1f1f"
INK_MUTED = "#5f5f5a"


def compute_metrics(y_true, y_pred, labels):
    idx = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]),
                   "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }


def plot_confusion(cm, labels, path, title):
    """Row-normalised confusion matrix; each cell shows its count and row share."""
    cm = np.asarray(cm)
    share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(1.4 * n + 2.2, 1.2 * n + 1.6), dpi=150)
    im = ax.imshow(share, cmap=CM_CMAP, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color="white" if share[i, j] > 0.55 else INK)

    ax.set_xticks(range(n), labels, rotation=30, ha="right", color=INK)
    ax.set_yticks(range(n), labels, color=INK)
    ax.set_xlabel("Predicted", color=INK_MUTED)
    ax.set_ylabel("True", color=INK_MUTED)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # thin surface-coloured gaps between cells
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share of true class", color=INK_MUTED)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
