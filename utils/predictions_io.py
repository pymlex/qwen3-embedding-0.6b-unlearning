from pathlib import Path

import numpy as np
import pandas as pd

from schemas import Config
from utils.plotting import plot_confusion_matrix


def save_test_predictions(
    config: Config,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    split_name: str,
    label_names: list[str],
) -> dict[str, str | int]:
    config.paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "label_id": y_true.astype(np.int64),
            "prediction_id": y_pred.astype(np.int64),
        }
    ).to_csv(config.paths.predictions_dir / f"{model_name}.csv", index=False)
    return {
        "model_name": model_name,
        "num_classes": num_classes,
        "split": split_name,
        "labels": "|".join(label_names),
    }


def write_predictions_manifest(config: Config, rows: list[dict[str, str | int]]) -> None:
    config.paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.paths.predictions_dir / "manifest.csv", index=False)


def predictions_source_dir(config: Config) -> Path:
    if (config.paths.predictions_dir / "manifest.csv").exists():
        return config.paths.predictions_dir
    return config.paths.results_predictions_dir


def replot_confusion_matrices(
    config: Config,
    predictions_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> list[Path]:
    source_dir = predictions_dir or predictions_source_dir(config)
    target_dir = figures_dir or config.paths.figures_dir
    manifest = pd.read_csv(source_dir / "manifest.csv")
    saved_paths: list[Path] = []
    target_dir.mkdir(parents=True, exist_ok=True)

    for _, row in manifest.iterrows():
        model_name = str(row["model_name"])
        label_names = str(row["labels"]).split("|")
        frame = pd.read_csv(source_dir / f"{model_name}.csv")
        figure_path = target_dir / f"confusion_{model_name}.png"
        plot_confusion_matrix(
            frame["label_id"].to_numpy(dtype=np.int64),
            frame["prediction_id"].to_numpy(dtype=np.int64),
            labels=label_names,
            title=f"Confusion matrix: {model_name}",
            save_path=figure_path,
        )
        saved_paths.append(figure_path)

    return saved_paths
