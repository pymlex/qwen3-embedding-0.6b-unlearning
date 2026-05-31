import os
import shutil
import subprocess

from utils.env_utils import ensure_env_file, load_env


GH_INSTALL_SCRIPT = """
set -e
if command -v gh >/dev/null 2>&1; then
  exit 0
fi
if command -v apt-get >/dev/null 2>&1; then
  type -p wget >/dev/null || (apt-get update && apt-get install -y wget)
  mkdir -p -m 755 /etc/apt/keyrings
  wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
  chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
  apt-get update
  apt-get install -y gh
fi
"""


def install_github_cli() -> None:
    if shutil.which("gh") is not None:
        return
    subprocess.run(["bash", "-c", GH_INSTALL_SCRIPT], check=True)


def login_github_browser() -> None:
    install_github_cli()
    subprocess.run(
        ["gh", "auth", "login", "-h", "github.com", "-p", "https", "-w"],
        check=True,
    )


def login_huggingface() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is empty in .env, skip Hugging Face login")
        return
    from huggingface_hub import login

    login(token=token)


def run_setup() -> None:
    ensure_env_file()
    load_env()
    login_github_browser()
    login_huggingface()
    print("auth setup finished")
