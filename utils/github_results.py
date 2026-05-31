import os
import shutil
import subprocess
from pathlib import Path

from schemas import Config


METRIC_FILENAMES = (
    "original_metrics.csv",
    "gold_metrics.csv",
    "final_evaluation.csv",
)

UNLEARNING_METRIC_SUFFIX = "_metrics.csv"


def metric_csv_paths(output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for name in METRIC_FILENAMES:
        path = output_dir / name
        if path.exists():
            paths.append(path)
    for path in sorted(output_dir.glob(f"*{UNLEARNING_METRIC_SUFFIX}")):
        if path.name in METRIC_FILENAMES:
            continue
        paths.append(path)
    return paths


def refresh_training_figures(config: Config) -> None:
    import pandas as pd

    from training.trainer import plot_all_training_curves

    gold_metrics = pd.read_csv(config.paths.output_dir / "gold_metrics.csv")
    original_metrics = pd.read_csv(config.paths.output_dir / "original_metrics.csv")
    histories: dict[str, pd.DataFrame] = {}
    for path in sorted(config.paths.output_dir.glob(f"*{UNLEARNING_METRIC_SUFFIX}")):
        if path.name in {"original_metrics.csv", "gold_metrics.csv"}:
            continue
        method_name = path.name.removesuffix(UNLEARNING_METRIC_SUFFIX)
        histories[method_name] = pd.read_csv(path)
    plot_all_training_curves(config, gold_metrics, original_metrics, histories)


def sync_results_artifacts(config: Config) -> tuple[list[Path], list[Path]]:
    """Copy metric tables and figures into tracked results/ for GitHub."""
    metrics_dir = config.paths.results_metrics_dir
    figures_dir = config.paths.results_figures_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    synced_metrics: list[Path] = []
    for source_path in metric_csv_paths(config.paths.output_dir):
        target_path = metrics_dir / source_path.name
        shutil.copy2(source_path, target_path)
        synced_metrics.append(target_path)

    synced_figures: list[Path] = []
    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(config.paths.figures_dir.glob("*.png")):
        target_path = figures_dir / source_path.name
        shutil.copy2(source_path, target_path)
        synced_figures.append(target_path)

    token_histogram = config.paths.repo_figures_dir / "token_length_distribution.png"
    if token_histogram.exists():
        target_path = figures_dir / token_histogram.name
        shutil.copy2(token_histogram, target_path)
        synced_figures.append(target_path)

    return synced_metrics, synced_figures


def git_push_with_token() -> None:
    token = os.environ.get("GH_TOKEN", "").strip()
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if token:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        repo_path = remote.removeprefix("https://github.com/").removeprefix("git@github.com:")
        repo_path = repo_path.removesuffix(".git")
        push_url = f"https://x-access-token:{token}@github.com/{repo_path}.git"
        subprocess.run(["git", "push", push_url, branch], check=True)
        return
    subprocess.run(["git", "push", "origin", branch], check=True)


def push_results_to_github(
    config: Config,
    commit_message: str = "Update experiment metrics and figures",
    refresh_figures: bool = True,
) -> None:
    """Sync non-checkpoint artifacts to results/ and push them to GitHub."""
    if refresh_figures:
        refresh_training_figures(config)

    synced_metrics, synced_figures = sync_results_artifacts(config)
    if len(synced_metrics) == 0 and len(synced_figures) == 0:
        print("no metric CSV or figure files found under outputs/")
        return

    subprocess.run(["git", "add", str(config.paths.results_dir)], check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", str(config.paths.results_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        print("results/ already matches outputs/, nothing to commit")
        return

    subprocess.run(["git", "commit", "-m", commit_message], check=True)
    git_push_with_token()
    print(
        f"pushed {len(synced_metrics)} metric tables and "
        f"{len(synced_figures)} figures to GitHub"
    )
