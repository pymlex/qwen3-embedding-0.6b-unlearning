import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def convert_tab_csv_to_comma(source_path: Path, target_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_path, sep="\t", quoting=csv.QUOTE_MINIMAL)
    frame = frame.rename(columns=str.strip)
    frame.to_csv(target_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return frame


def load_comma_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, quoting=csv.QUOTE_MINIMAL)


def compute_token_lengths(
    texts: list[str],
    model_id: str,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    lengths = []
    for start in tqdm(range(0, len(texts), batch_size), desc="token lengths"):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(batch_texts, add_special_tokens=True, truncation=False)
        batch_lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
        lengths.extend(batch_lengths)
    return np.asarray(lengths, dtype=np.int64)


def plot_token_length_histogram(
    lengths: np.ndarray,
    p99: float,
    max_length: int,
    save_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(lengths, bins=80, color="#4C72B0", edgecolor="white")
    axis.axvline(p99, color="#C44E52", linestyle="--", linewidth=1.5, label=f"p99 = {int(p99)}")
    axis.axvline(max_length, color="#55A868", linestyle="-", linewidth=1.5, label=f"max_length = {max_length}")
    axis.set_xlabel("Token count (Qwen3-Embedding-0.6B tokenizer)")
    axis.set_ylabel("Reviews")
    axis.set_title("Review length distribution")
    axis.legend()
    axis.grid(alpha=0.5)
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150)
    plt.close(figure)


def analyze_dataset(
    csv_path: Path,
    model_id: str,
    figures_dir: Path,
    stats_path: Path,
    batch_size: int = 256,
) -> dict[str, float | int]:
    frame = load_comma_csv(csv_path)
    texts = frame["review"].astype(str).tolist()
    lengths = compute_token_lengths(texts, model_id, batch_size)
    p99 = float(np.percentile(lengths, 99))
    max_length = 256
    stats = {
        "num_reviews": int(len(lengths)),
        "mean_tokens": float(np.mean(lengths)),
        "median_tokens": float(np.median(lengths)),
        "p95_tokens": float(np.percentile(lengths, 95)),
        "p99_tokens": p99,
        "max_tokens": int(np.max(lengths)),
        "max_length": max_length,
    }
    plot_token_length_histogram(
        lengths,
        p99=p99,
        max_length=max_length,
        save_path=figures_dir / "token_length_distribution.png",
    )
    pd.DataFrame({"token_length": lengths}).to_csv(stats_path.with_suffix(".csv"), index=False)
    stats_path.write_text(
        "\n".join(f"{key}={value}" for key, value in stats.items()),
        encoding="utf-8",
    )
    return stats
