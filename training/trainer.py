import copy
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from constants import ID2LABEL, LABEL2ID, RETAIN_CLASSES
from data.dataset import PairDataset, ReviewDataset, collate_pair_batch, collate_review_batch
from metrics.evaluation import evaluate_split, evaluate_unlearning_metrics
from models.classifier import QwenEmbeddingClassifier
from schemas import Config
from training.losses import dpo_like_loss, random_target_loss, retain_ft_loss, rmu_loss
from utils.mlflow_utils import log_metrics_dict, start_run
from utils.plotting import plot_confusion_matrix, plot_training_curves


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(config: Config) -> QwenEmbeddingClassifier:
    return QwenEmbeddingClassifier(
        model_id=config.model.base_model_id,
        num_classes=config.model.num_classes,
        hidden_dim=config.model.mlp_hidden_dim,
        max_length=config.model.max_length,
    )


def _parameter_groups(model: QwenEmbeddingClassifier, config: Config) -> list[dict[str, object]]:
    return [
        {"params": model.encoder.parameters(), "lr": config.train.learning_rate},
        {"params": model.classifier.parameters(), "lr": config.train.head_learning_rate},
    ]


def _eval_steps_per_epoch(steps_per_epoch: int, eval_interval_epochs: float) -> int:
    return max(1, int(steps_per_epoch * eval_interval_epochs))


def _epoch_from_step(step: int, steps_per_epoch: int) -> float:
    return step / steps_per_epoch


def _collect_baseline_metrics(
    model: QwenEmbeddingClassifier,
    splits: dict[str, pd.DataFrame],
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    valid_texts = splits["valid"]["review"].astype(str).tolist()
    valid_labels = splits["valid"]["label_id"].astype(int).tolist()
    valid_eval = evaluate_split(model, valid_texts, valid_labels, device, config.train.batch_size)
    return {"valid_mcc": valid_eval["mcc"]}


def train_baseline_model(
    config: Config,
    splits: dict[str, pd.DataFrame],
    run_name: str,
    save_name: str,
) -> tuple[QwenEmbeddingClassifier, pd.DataFrame]:
    device = _device()
    model = _build_model(config).to(device)
    train_dataset = ReviewDataset(splits["train"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate_review_batch,
    )

    optimizer = torch.optim.AdamW(
        _parameter_groups(model, config),
        weight_decay=config.train.weight_decay,
    )
    total_steps = int(len(train_loader) * config.train.epochs / config.train.gradient_accumulation_steps)
    warmup_steps = int(total_steps * config.train.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    metric_rows = []
    global_step = 0
    steps_per_epoch = len(train_loader)
    eval_every = _eval_steps_per_epoch(steps_per_epoch, config.train.eval_interval_epochs)
    total_optimizer_steps = int(config.train.epochs * steps_per_epoch / config.train.gradient_accumulation_steps)

    with start_run(config, run_name):
        epoch_zero_metrics = _collect_baseline_metrics(model, splits, config, device)
        epoch_zero_metrics["epoch"] = 0.0
        metric_rows.append(epoch_zero_metrics)
        log_metrics_dict(epoch_zero_metrics, step=0)
        print(f"epoch=0.0 valid_mcc={epoch_zero_metrics['valid_mcc']:.4f}")

        progress = tqdm(total=total_optimizer_steps, desc=f"train {save_name}")
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            logits = model(batch["texts"], device)
            loss = retain_ft_loss(logits, batch["labels"].to(device))
            loss = loss / config.train.gradient_accumulation_steps
            loss.backward()

            if (batch_index + 1) % config.train.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)

                if global_step % eval_every == 0 or global_step == total_optimizer_steps:
                    current_epoch = _epoch_from_step(global_step * config.train.gradient_accumulation_steps, steps_per_epoch)
                    metrics = _collect_baseline_metrics(model, splits, config, device)
                    metrics["epoch"] = current_epoch
                    metrics["train_loss"] = float(loss.item() * config.train.gradient_accumulation_steps)
                    metric_rows.append(metrics)
                    log_metrics_dict(metrics, step=global_step)
                    print(f"epoch={current_epoch:.2f} valid_mcc={metrics['valid_mcc']:.4f}")

        progress.close()

        save_dir = config.paths.checkpoints_dir / save_name
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        mlflow.log_param("checkpoint_dir", str(save_dir))

    return model, pd.DataFrame(metric_rows)


def _retain_label_ids() -> list[int]:
    return [LABEL2ID[label] for label in RETAIN_CLASSES]


def run_unlearning_method(
    config: Config,
    splits: dict[str, pd.DataFrame],
    gold_model: QwenEmbeddingClassifier,
    method: str,
) -> tuple[QwenEmbeddingClassifier, pd.DataFrame]:
    device = _device()
    original_dir = config.paths.checkpoints_dir / "original"
    model = QwenEmbeddingClassifier.load_pretrained(
        str(original_dir),
        config.model.base_model_id,
        config.model.num_classes,
        config.model.mlp_hidden_dim,
        config.model.max_length,
    ).to(device)
    reference_model = copy.deepcopy(model)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False

    gold_model = gold_model.to(device)
    gold_model.eval()

    pair_dataset = PairDataset(splits["retain_train"], splits["forget_train"])
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate_pair_batch,
    )

    optimizer = torch.optim.AdamW(
        _parameter_groups(model, config),
        lr=config.train.unlearning_learning_rate,
        weight_decay=config.train.weight_decay,
    )

    retain_test_texts = splits["retain_test"]["review"].astype(str).tolist()
    retain_test_labels = splits["retain_test"]["label_id"].astype(int).tolist()
    forget_test_texts = splits["forget_test"]["review"].astype(str).tolist()
    forget_test_labels = splits["forget_test"]["label_id"].astype(int).tolist()
    retain_label_ids = _retain_label_ids()

    metric_rows = []
    global_step = 0
    steps_per_epoch = len(pair_loader)
    total_optimizer_steps = int(config.train.unlearning_epochs * steps_per_epoch / config.train.gradient_accumulation_steps)
    eval_every = _eval_steps_per_epoch(steps_per_epoch, config.train.eval_interval_epochs)

    with start_run(config, f"unlearn_{method}"):
        epoch_zero = evaluate_unlearning_metrics(
            model,
            gold_model,
            retain_test_texts,
            retain_test_labels,
            forget_test_texts,
            forget_test_labels,
            device,
            config.train.batch_size,
        )
        epoch_zero["epoch"] = 0.0
        metric_rows.append(epoch_zero)
        log_metrics_dict(epoch_zero, step=0)
        print(
            f"method={method} epoch=0.0 "
            f"retain_mcc={epoch_zero['model_retain_mcc']:.4f} "
            f"forget_mcc={epoch_zero['model_forget_mcc']:.4f}"
        )

        progress = tqdm(total=total_optimizer_steps, desc=f"unlearn {method}")
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(pair_loader):
            retain_logits = model(batch["retain_texts"], device)
            forget_logits = model(batch["forget_texts"], device)
            retain_labels = batch["retain_labels"].to(device)
            forget_labels = batch["forget_labels"].to(device)

            if method == "retain_ft":
                loss = retain_ft_loss(retain_logits, retain_labels)
            elif method == "dpo_like":
                with torch.no_grad():
                    reference_retain_logits = reference_model(batch["retain_texts"], device)
                    reference_forget_logits = reference_model(batch["forget_texts"], device)
                loss = dpo_like_loss(
                    retain_logits,
                    retain_labels,
                    forget_logits,
                    forget_labels,
                    reference_retain_logits,
                    reference_forget_logits,
                    config.train.dpo_beta,
                )
            elif method == "rmu":
                with torch.no_grad():
                    reference_retain_logits = reference_model(batch["retain_texts"], device)
                loss = rmu_loss(
                    retain_logits,
                    retain_labels,
                    forget_logits,
                    reference_retain_logits,
                    config.model.num_classes,
                )
            elif method == "random_target":
                random_indices = torch.randint(
                    low=0,
                    high=len(retain_label_ids),
                    size=(forget_labels.shape[0],),
                    device=device,
                )
                random_labels = torch.tensor(retain_label_ids, device=device)[random_indices]
                loss = random_target_loss(
                    retain_logits,
                    retain_labels,
                    forget_logits,
                    random_labels,
                    config.train.random_target_gamma,
                )

            loss = loss / config.train.gradient_accumulation_steps
            loss.backward()

            if (batch_index + 1) % config.train.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)

                if global_step % eval_every == 0 or global_step == total_optimizer_steps:
                    current_epoch = _epoch_from_step(
                        global_step * config.train.gradient_accumulation_steps,
                        steps_per_epoch,
                    )
                    metrics = evaluate_unlearning_metrics(
                        model,
                        gold_model,
                        retain_test_texts,
                        retain_test_labels,
                        forget_test_texts,
                        forget_test_labels,
                        device,
                        config.train.batch_size,
                    )
                    metrics["epoch"] = current_epoch
                    metrics["train_loss"] = float(loss.item() * config.train.gradient_accumulation_steps)
                    metric_rows.append(metrics)
                    log_metrics_dict(metrics, step=global_step)
                    print(
                        f"method={method} epoch={current_epoch:.2f} "
                        f"retain_mcc={metrics['model_retain_mcc']:.4f} "
                        f"forget_mcc={metrics['model_forget_mcc']:.4f}"
                    )

        progress.close()

        save_dir = config.paths.checkpoints_dir / "unlearn" / method
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        mlflow.log_param("checkpoint_dir", str(save_dir))

    return model, pd.DataFrame(metric_rows)


def evaluate_test_and_plot(
    config: Config,
    model: QwenEmbeddingClassifier,
    splits: dict[str, pd.DataFrame],
    model_name: str,
) -> dict[str, float]:
    device = _device()
    model = model.to(device)
    test_texts = splits["test"]["review"].astype(str).tolist()
    test_labels = splits["test"]["label_id"].astype(int).tolist()
    evaluation = evaluate_split(model, test_texts, test_labels, device, config.train.batch_size)
    confusion = evaluation["labels"], evaluation["predictions"]
    figure_path = config.paths.figures_dir / f"confusion_{model_name}.png"
    plot_confusion_matrix(
        confusion[0],
        confusion[1],
        labels=[ID2LABEL[index] for index in range(config.model.num_classes)],
        title=f"Confusion matrix: {model_name}",
        save_path=figure_path,
    )
    return {"test_mcc": evaluation["mcc"]}


def select_best_unlearning_method(summary_df: pd.DataFrame, gold_retain_mcc: float) -> str:
    threshold = 0.9 * gold_retain_mcc
    eligible = summary_df[summary_df["model_retain_mcc"] >= threshold]
    if len(eligible) == 0:
        summary_df = summary_df.copy()
        summary_df["score"] = summary_df["model_retain_mcc"] - summary_df["model_forget_mcc"]
        return str(summary_df.sort_values("score", ascending=False).iloc[0]["method"])
    return str(eligible.sort_values("model_forget_mcc", ascending=True).iloc[0]["method"])


def save_metric_tables(config: Config, baseline_df: pd.DataFrame, unlearning_summary: pd.DataFrame) -> None:
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(config.paths.output_dir / "baseline_metrics.csv", index=False)
    unlearning_summary.to_csv(config.paths.output_dir / "unlearning_summary.csv", index=False)


def plot_all_training_curves(
    config: Config,
    gold_df: pd.DataFrame,
    original_df: pd.DataFrame,
    unlearning_histories: dict[str, pd.DataFrame],
) -> None:
    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        {"gold": gold_df, "original": original_df},
        metric="valid_mcc",
        title="Baseline validation MCC",
        save_path=config.paths.figures_dir / "baseline_valid_mcc.png",
    )
    for method, frame in unlearning_histories.items():
        plot_training_curves(
            {method: frame},
            metric="model_retain_mcc",
            title=f"{method} retain MCC",
            save_path=config.paths.figures_dir / f"{method}_retain_mcc.png",
        )
        plot_training_curves(
            {method: frame},
            metric="model_forget_mcc",
            title=f"{method} forget MCC",
            save_path=config.paths.figures_dir / f"{method}_forget_mcc.png",
        )
