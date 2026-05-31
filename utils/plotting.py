from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    save_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150)
    plt.close(figure)


def plot_training_curves(
    histories: dict[str, pd.DataFrame],
    metric: str,
    title: str,
    save_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, frame in histories.items():
        axis.plot(frame["epoch"], frame[metric], marker="o", label=name)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.5)
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
