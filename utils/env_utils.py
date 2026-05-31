import os
import shutil
from pathlib import Path


def ensure_env_file(
    example_path: Path = Path(".env.example"),
    env_path: Path = Path(".env"),
) -> Path:
    if not env_path.exists():
        shutil.copy(example_path, env_path)
        print(f"created {env_path} from {example_path}")
    return env_path


def load_env(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if value:
            os.environ[key] = value
    gh_token = os.environ.get("GH_TOKEN", "").strip()
    if gh_token:
        os.environ["GITHUB_TOKEN"] = gh_token
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
