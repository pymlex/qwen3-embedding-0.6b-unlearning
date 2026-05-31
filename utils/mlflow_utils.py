from contextlib import contextmanager
from typing import Iterator

import mlflow

from schemas import Config


def setup_mlflow(config: Config) -> None:
    mlflow.set_experiment(config.paths.mlflow_experiment)


@contextmanager
def start_run(config: Config, run_name: str) -> Iterator[None]:
    setup_mlflow(config)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config.model.model_dump())
        mlflow.log_params(config.train.model_dump())
        mlflow.log_params(config.data.model_dump())
        yield


def log_metrics_dict(metrics: dict[str, float], step: int) -> None:
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)
