"""Common/plotting.py - saves diagnostic figures to the Figure/ directory."""

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Config import config
from Common.utils import ensure_dir, get_logger

logger = get_logger("plotting")


def _savefig(fig, name: str):
    ensure_dir(config.FIGURE_DIR)
    out_path = config.FIGURE_DIR / f"{name}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")
    return out_path


def plot_model_comparison(leaderboard: pd.DataFrame, name: str = "model_comparison"):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(leaderboard))))
    ax.barh(leaderboard["model"], leaderboard["mean_score"], xerr=leaderboard["std_score"])
    ax.set_xlabel(leaderboard["scoring"].iloc[0] if not leaderboard.empty else "score")
    ax.set_title("Model comparison")
    ax.invert_yaxis()
    return _savefig(fig, name)


def plot_confusion_matrix(cm: np.ndarray, labels=None, name: str = "confusion_matrix"):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    return _savefig(fig, name)


def plot_feature_importance(model, feature_names, top_n: int = 20, name: str = "feature_importance"):
    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{model.__class__.__name__} has no feature_importances_; skipping plot.")
        return None
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    importances = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(importances))))
    ax.barh(importances.index, importances.values)
    ax.invert_yaxis()
    ax.set_title("Feature importance")
    return _savefig(fig, name)
