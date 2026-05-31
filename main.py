import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


UNLEARNING_METHODS = ("retain_ft", "dpo_like", "rmu", "random_target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-Embedding machine unlearning pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("setup", help="Create .env, GitHub browser login, Hugging Face login")

    subparsers.add_parser("analyze-dataset", help="Convert CSV separator, compute p99 token length, plot histogram")

    subparsers.add_parser("prepare-data", help="Create train, valid, test and retain or forget splits")

    subparsers.add_parser("train-baseline", help="Train gold and original models for two epochs")

    unlearn_parser = subparsers.add_parser("unlearn", help="Run one or all unlearning methods")
    unlearn_parser.add_argument(
        "--method",
        choices=[*UNLEARNING_METHODS, "all"],
        default="all",
    )

    subparsers.add_parser("evaluate", help="Evaluate all checkpoints and build confusion matrices")

    push_parser = subparsers.add_parser("push-hf", help="Upload checkpoints to Hugging Face Hub")
    push_parser.add_argument("--token", default=None)

    subparsers.add_parser("run-all", help="Prepare data, train baseline, unlearn, evaluate, push to HF")

    return parser.parse_args()


def load_splits(config) -> dict[str, pd.DataFrame]:
    split_files = {
        "train": config.paths.splits_dir / "train.csv",
        "valid": config.paths.splits_dir / "valid.csv",
        "test": config.paths.splits_dir / "test.csv",
        "retain_train": config.paths.splits_dir / "retain_train.csv",
        "forget_train": config.paths.splits_dir / "forget_train.csv",
        "retain_test": config.paths.splits_dir / "retain_test.csv",
        "forget_test": config.paths.splits_dir / "forget_test.csv",
    }
    return {name: pd.read_csv(path) for name, path in split_files.items()}


def command_analyze_dataset(config) -> dict[str, float | int]:
    from data.token_stats import analyze_dataset, convert_tab_csv_to_comma, load_comma_csv

    csv_path = config.data.csv_path
    first_line = csv_path.read_text(encoding="utf-8").splitlines()[0]
    if "\t" in first_line and first_line.count(",") == 0:
        convert_tab_csv_to_comma(csv_path, csv_path)
        print(f"converted {csv_path} to comma-separated format")
    else:
        load_comma_csv(csv_path)
        print(f"{csv_path} already uses comma separator")

    stats = analyze_dataset(
        csv_path=csv_path,
        model_id=config.model.base_model_id,
        figures_dir=config.paths.repo_figures_dir,
        stats_path=config.paths.token_stats_path,
    )
    config.apply_token_stats()
    print(
        f"p99={stats['p99_tokens']:.0f} max_length={stats['max_length']} "
        f"mean={stats['mean_tokens']:.1f} median={stats['median_tokens']:.1f}"
    )
    return stats


def command_prepare_data(config) -> dict[str, pd.DataFrame]:
    from data.splits import prepare_splits

    splits = prepare_splits(config)
    print(
        f"train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])} "
        f"retain_train={len(splits['retain_train'])} forget_train={len(splits['forget_train'])}"
    )
    return splits


def command_train_baseline(config, splits: dict[str, pd.DataFrame]) -> None:
    from data.splits import frame_with_retain_labels
    from training.trainer import plot_all_training_curves, train_baseline_model

    config.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    original_model, original_metrics = train_baseline_model(
        config,
        splits["train"],
        splits["valid"],
        run_name="original",
        save_name="original",
        num_classes=config.model.num_classes,
    )

    gold_train_df = frame_with_retain_labels(splits["retain_train"])
    gold_valid_df = frame_with_retain_labels(splits["valid"])
    gold_model, gold_metrics = train_baseline_model(
        config,
        gold_train_df,
        gold_valid_df,
        run_name="gold",
        save_name="gold",
        num_classes=config.model.gold_num_classes,
    )

    original_metrics.to_csv(config.paths.output_dir / "original_metrics.csv", index=False)
    gold_metrics.to_csv(config.paths.output_dir / "gold_metrics.csv", index=False)
    plot_all_training_curves(config, gold_metrics, original_metrics, {})
    print(f"original final valid_mcc={original_metrics.iloc[-1]['valid_mcc']:.4f}")
    print(f"gold final valid_mcc={gold_metrics.iloc[-1]['valid_mcc']:.4f}")


def command_unlearn(
    config,
    splits: dict[str, pd.DataFrame],
    method: str,
) -> dict[str, pd.DataFrame]:
    from models.classifier import QwenEmbeddingClassifier
    from training.trainer import run_unlearning_method

    gold_dir = config.paths.checkpoints_dir / "gold"
    gold_model = QwenEmbeddingClassifier.load_pretrained(
        str(gold_dir),
        config.model.base_model_id,
        config.model.gold_num_classes,
        config.model.mlp_hidden_dim,
        config.model.max_length,
    )
    methods = UNLEARNING_METHODS if method == "all" else (method,)
    histories = {}
    for unlearning_method in methods:
        _, history = run_unlearning_method(config, splits, gold_model, unlearning_method)
        history.to_csv(config.paths.output_dir / f"{unlearning_method}_metrics.csv", index=False)
        histories[unlearning_method] = history
    return histories


def command_evaluate(config, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    import torch

    from metrics.evaluation import evaluate_unlearning_metrics
    from models.classifier import QwenEmbeddingClassifier
    from training.trainer import evaluate_test_and_plot, select_best_unlearning_method

    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows = []

    gold_model = QwenEmbeddingClassifier.load_pretrained(
        str(config.paths.checkpoints_dir / "gold"),
        config.model.base_model_id,
        config.model.gold_num_classes,
        config.model.mlp_hidden_dim,
        config.model.max_length,
    )

    from data.splits import frame_with_retain_labels

    gold_test_df = frame_with_retain_labels(splits["retain_test"])

    for model_name, num_classes, test_df in (
        ("original", config.model.num_classes, splits["test"]),
        ("gold", config.model.gold_num_classes, gold_test_df),
    ):
        model = QwenEmbeddingClassifier.load_pretrained(
            str(config.paths.checkpoints_dir / model_name),
            config.model.base_model_id,
            num_classes,
            config.model.mlp_hidden_dim,
            config.model.max_length,
        )
        test_metrics = evaluate_test_and_plot(config, model, test_df, model_name, num_classes)
        unlearning_metrics = evaluate_unlearning_metrics(
            model,
            gold_model,
            splits["retain_test"]["review"].astype(str).tolist(),
            splits["retain_test"]["label_id"].astype(int).tolist(),
            splits["forget_test"]["review"].astype(str).tolist(),
            splits["forget_test"]["label_id"].astype(int).tolist(),
            device,
            config.train.batch_size,
        )
        row = {"method": model_name, **test_metrics, **unlearning_metrics}
        summary_rows.append(row)

    gold_retain_mcc = next(row["model_retain_mcc"] for row in summary_rows if row["method"] == "gold")
    unlearn_root = config.paths.checkpoints_dir / "unlearn"
    if unlearn_root.exists():
        for method_dir in sorted(unlearn_root.iterdir()):
            if not method_dir.is_dir():
                continue
            model = QwenEmbeddingClassifier.load_pretrained(
                str(method_dir),
                config.model.base_model_id,
                config.model.num_classes,
                config.model.mlp_hidden_dim,
                config.model.max_length,
            )
            test_metrics = evaluate_test_and_plot(
                config,
                model,
                splits["test"],
                f"unlearn_{method_dir.name}",
                config.model.num_classes,
            )
            unlearning_metrics = evaluate_unlearning_metrics(
                model,
                gold_model,
                splits["retain_test"]["review"].astype(str).tolist(),
                splits["retain_test"]["label_id"].astype(int).tolist(),
                splits["forget_test"]["review"].astype(str).tolist(),
                splits["forget_test"]["label_id"].astype(int).tolist(),
                device,
                config.train.batch_size,
            )
            summary_rows.append({"method": method_dir.name, **test_metrics, **unlearning_metrics})

    summary_df = pd.DataFrame(summary_rows)
    unlearning_only = summary_df[~summary_df["method"].isin(["gold", "original"])].copy()
    if len(unlearning_only) > 0:
        best_method = select_best_unlearning_method(unlearning_only, gold_retain_mcc)
        best_row = unlearning_only[unlearning_only["method"] == best_method].iloc[0]
        print(f"best unlearning method={best_method} test_mcc={best_row['test_mcc']:.4f}")
        best_source = config.paths.figures_dir / f"confusion_unlearn_{best_method}.png"
        best_target = config.paths.figures_dir / "confusion_best_unlearn.png"
        if best_source.exists():
            shutil.copy(best_source, best_target)

    summary_df.to_csv(config.paths.output_dir / "final_evaluation.csv", index=False)
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return summary_df


def main() -> None:
    from schemas import Config
    from utils.env_utils import ensure_env_file, load_env

    ensure_env_file()
    load_env()

    args = parse_args()
    config = Config()
    config.apply_token_stats()
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "setup":
        from utils.setup_auth import run_setup

        run_setup()
        return

    if args.command == "analyze-dataset":
        command_analyze_dataset(config)
        return

    if args.command == "prepare-data":
        command_prepare_data(config)
        return

    splits = load_splits(config) if config.paths.splits_dir.exists() else command_prepare_data(config)

    if args.command == "train-baseline":
        command_train_baseline(config, splits)
        return

    if args.command == "unlearn":
        command_unlearn(config, splits, args.method)
        return

    if args.command == "evaluate":
        command_evaluate(config, splits)
        return

    if args.command == "push-hf":
        from utils.hf_upload import push_all_models

        token = args.token or os.environ.get("HF_TOKEN")
        push_all_models(config, token=token)
        return

    if args.command == "run-all":
        command_analyze_dataset(config)
        splits = command_prepare_data(config)
        command_train_baseline(config, splits)
        histories = command_unlearn(config, splits, "all")
        from training.trainer import plot_all_training_curves

        gold_metrics = pd.read_csv(config.paths.output_dir / "gold_metrics.csv")
        plot_all_training_curves(config, gold_metrics, gold_metrics, histories)
        command_evaluate(config, splits)
        from utils.hf_upload import push_all_models

        push_all_models(config, token=os.environ.get("HF_TOKEN"))
        return


if __name__ == "__main__":
    main()
