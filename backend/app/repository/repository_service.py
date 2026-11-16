
import asyncio
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse


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

    process = await asyncio.create_subprocess_exec(
        "git",
        "clone",
        "--depth", "1",
        "--single-branch",
        clone_url,
        str(workspace),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        _, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=CLONE_TIMEOUT
        )

    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        shutil.rmtree(workspace, ignore_errors=True)

        raise RepositoryError("Repository cloning timed out.")

    if process.returncode != 0:
        shutil.rmtree(workspace, ignore_errors=True)

        raise RepositoryError(
            stderr.decode(errors="replace")[-2000:]
        )

    return str(workspace)
