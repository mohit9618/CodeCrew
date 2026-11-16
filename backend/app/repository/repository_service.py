
import asyncio
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse
import subprocess


WORKSPACE_ROOT = Path("workspace").resolve()
CLONE_TIMEOUT = 180


class RepositoryError(Exception):
    pass


def validate_github_url(repo_url: str) -> str:
    parsed = urlparse(repo_url.strip())

    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.port not in (None, 443)
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RepositoryError(
            "Only standard HTTPS GitHub repository URLs are supported."
        )

    parts = parsed.path.strip("/").split("/")

    if len(parts) != 2:
        raise RepositoryError("Invalid GitHub repository URL.")

    owner, repo = parts
    repo = repo.removesuffix(".git")

    if not re.fullmatch(r"[A-Za-z0-9-]+", owner):
        raise RepositoryError("Invalid GitHub owner.")

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        raise RepositoryError("Invalid repository name.")

    return f"https://github.com/{owner}/{repo}.git"



async def clone_repository(repo_url: str, run_id: str) -> str:
    clone_url = validate_github_url(repo_url)

    if not re.fullmatch(r"[a-fA-F0-9-]{36}", run_id):
        raise RepositoryError("Invalid run ID.")

    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    workspace = (WORKSPACE_ROOT / run_id).resolve()

    if not workspace.is_relative_to(WORKSPACE_ROOT):
        raise RepositoryError("Invalid workspace path.")

    if workspace.exists():
        raise RepositoryError("Workspace already exists.")

    try:
        process = await asyncio.to_thread(
            subprocess.run,
            [
                "git",
                "clone",
                "--depth", "1",
                "--single-branch",
                clone_url,
                str(workspace)
            ],
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT,
            check=False
        )

    except subprocess.TimeoutExpired:
        shutil.rmtree(workspace, ignore_errors=True)
        raise RepositoryError("Repository cloning timed out.")

    except FileNotFoundError:
        shutil.rmtree(workspace, ignore_errors=True)
        raise RepositoryError(
            "Git executable not found. Install Git and add it to PATH."
        )

    if process.returncode != 0:
        shutil.rmtree(workspace, ignore_errors=True)
        raise RepositoryError(process.stderr[-2000:])

    return str(workspace)
