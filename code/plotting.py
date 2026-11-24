"""All figures."""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from config import ASSUMPTION_FEATURES, FEATURES, GROUP_ORDER, RNG


def plot_features(st, path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, f in zip(axes.ravel(), FEATURES):
        data = [st.loc[st.group == g, f].dropna() for g in GROUP_ORDER]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels(GROUP_ORDER)
        for i, v in enumerate(data, 1):
            ax.scatter(RNG.normal(i, 0.06, len(v)), v, s=14, alpha=.6)
        if f in ASSUMPTION_FEATURES:
            ax.axhline(0, color="gray", ls="--", lw=.8)
        ax.set_title(f); ax.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_variance_of_sums(st, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in [(axes[0], "pair_ratio", "Var(X_i+X_{i+1}) / (2 Var X)"),
                           (axes[1], "block_ratio", "Var(10-stride sum) / (10 Var X)")]:
        data = [st.loc[st.group == g, col].dropna() for g in GROUP_ORDER]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels(GROUP_ORDER)
        for i, v in enumerate(data, 1):
            ax.scatter(RNG.normal(i, 0.06, len(v)), v, s=14, alpha=.6)
        ax.axhline(1, color="red", ls="--", lw=1, label="independence prediction")
        ax.set_title(title); ax.legend(); ax.tick_params(axis="x", rotation=20)
        ax.set_yscale("log")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_confusions(y, preds, path):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4))
    for ax, (name, p) in zip(np.atleast_1d(axes), preds.items()):
        cm = confusion_matrix(y, p, labels=GROUP_ORDER)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_xticklabels(GROUP_ORDER, rotation=45, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(GROUP_ORDER)
        for i, j in itertools.product(range(4), range(4)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(name, fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
