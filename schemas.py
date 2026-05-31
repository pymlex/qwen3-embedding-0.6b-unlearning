from pathlib import Path

from pydantic import BaseModel, Field


def load_max_length_from_stats(stats_path: Path = Path("dataset_token_stats.json")) -> int:
    lines = stats_path.read_text(encoding="utf-8").strip().splitlines()
    values = dict(line.split("=", maxsplit=1) for line in lines)
    return int(float(values["max_length"]))


class DataConfig(BaseModel):
    csv_path: Path = Path("women_clothing_accessories.csv")
    test_per_class: int = 1000
    valid_per_class: int = 1000
    train_per_class: int = 8000
    forget_class: str = "neutral"
    seed: int = 42


class ModelConfig(BaseModel):
    base_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    num_classes: int = 3
    gold_num_classes: int = 2
    mlp_hidden_dim: int = 512
    max_length: int = 128
    embedding_dim: int = 1024


class TrainConfig(BaseModel):
    epochs: float = 1.0
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_interval_epochs: float = 0.1
    unlearning_epochs: float = 1.0
    unlearning_learning_rate: float = 1e-5
    dpo_beta: float = 1.0
    random_target_gamma: float = 0.7
    max_grad_norm: float = 1.0


class PathConfig(BaseModel):
    output_dir: Path = Path("outputs")
    checkpoints_dir: Path = Path("outputs/checkpoints")
    figures_dir: Path = Path("outputs/figures")
    repo_figures_dir: Path = Path("figures")
    results_dir: Path = Path("results")
    results_metrics_dir: Path = Path("results/metrics")
    results_figures_dir: Path = Path("results/figures")
    results_predictions_dir: Path = Path("results/predictions")
    predictions_dir: Path = Path("outputs/predictions")
    splits_dir: Path = Path("outputs/splits")
    token_stats_path: Path = Path("dataset_token_stats.json")
    mlflow_experiment: str = "qwen3-embedding-unlearning"
    hf_repo_id: str = "pymlex/qwen3-embedding-0.6b-unlearning"


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    def apply_token_stats(self) -> None:
        if self.paths.token_stats_path.exists():
            self.model.max_length = load_max_length_from_stats(self.paths.token_stats_path)
