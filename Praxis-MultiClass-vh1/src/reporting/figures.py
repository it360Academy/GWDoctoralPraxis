"""Report figures: confusion matrix heatmap, per-class AUROC bar chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..features.definitions import DISEASE_ORDER


def plot_confusion_matrix(cm: pd.DataFrame, out_path: Path, normalize: bool = True) -> Path:
    mat = cm.to_numpy().astype(np.float64)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat = mat / row_sums
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(DISEASE_ORDER)))
    ax.set_yticks(range(len(DISEASE_ORDER)))
    ax.set_xticklabels(DISEASE_ORDER, rotation=90)
    ax.set_yticklabels(DISEASE_ORDER)
    ax.set_xlabel("Predicted DX1")
    ax.set_ylabel("True DX1")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_class_auroc(per_class: pd.DataFrame, out_path: Path) -> Path:
    df = per_class.set_index("phenotype").reindex(DISEASE_ORDER)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df.index, df["auroc_ovr"], color="#3b6fa0")
    ax.axvline(0.5, color="gray", linestyle="--", label="chance (0.5)")
    ax.set_xlabel("AUROC (one-vs-rest)")
    ax.set_title("Per-phenotype discrimination, test year")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_algorithm_comparison(model_perf: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    piv = model_perf.pivot_table(index="algorithm", columns="year_role", values="macro_auroc_ovr")
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("Macro AUROC-OVR")
    ax.set_title("Algorithm comparison across years")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
