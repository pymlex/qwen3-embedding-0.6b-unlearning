from pathlib import Path

from huggingface_hub import HfApi
from tqdm.auto import tqdm

from schemas import Config


def upload_checkpoint(
    local_dir: Path,
    repo_id: str,
    path_in_repo: str,
    token: str | None = None,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type="model",
    )


def push_all_models(config: Config, token: str | None = None) -> None:
    repo_id = config.paths.hf_repo_id
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    model_card = Path("hf_model_card.md")
    if model_card.exists():
        api.upload_file(
            path_or_fileobj=str(model_card),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )

    uploads = [
        (config.paths.checkpoints_dir / "gold", "gold"),
        (config.paths.checkpoints_dir / "original", "original"),
    ]
    unlearn_root = config.paths.checkpoints_dir / "unlearn"
    if unlearn_root.exists():
        for method_dir in unlearn_root.iterdir():
            if method_dir.is_dir():
                uploads.append((method_dir, f"unlearn/{method_dir.name}"))

    for local_dir, remote_path in tqdm(uploads, desc="push Hugging Face"):
        upload_checkpoint(local_dir, repo_id, remote_path, token=token)
