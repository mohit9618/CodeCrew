"""Publish approved changes: branch, commit, push and open a pull request.

Runs only after a human approved the result. Needs a GitHub token with
write access to the repository (GITHUB_TOKEN). When the token is missing
or the push is refused, the changes stay in the workspace and the reason is
reported; nothing here raises into the workflow.

The token is passed to git through environment variables, never on the
command line, so it does not show up in process lists or logs.
"""
import base64
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from backend.app.utils.paths import safe_relative_path

GIT_TIMEOUT = 120
API_TIMEOUT = 30
GITHUB_API = "https://api.github.com"

DEFAULT_GIT_NAME = "CodeCrew"
DEFAULT_GIT_EMAIL = "codecrew@users.noreply.github.com"


class PublishError(Exception):
    pass


def slugify(text: str, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

    return slug[:max_length].strip("-") or "change"


def build_branch_name(request: str, workspace: str) -> str:
    """codecrew/<short-request>-<run id prefix>, unique per run."""
    run_id = Path(workspace).name.replace("-", "")[:8] or "run"

    return f"codecrew/{slugify(request)}-{run_id}"


def parse_github_repo(remote_url: str) -> tuple[str, str]:
    match = re.search(
        r"github\.com[/:]([A-Za-z0-9-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$",
        remote_url.strip(),
    )

    if not match:
        raise PublishError(f"Not a GitHub repository URL: {remote_url}")

    return match.group(1), match.group(2)


def _redact(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


def _git_env(token: str) -> dict:
    """Environment that gives git the token without putting it in argv."""
    credentials = base64.b64encode(
        f"x-access-token:{token}".encode()
    ).decode()

    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {credentials}",
    }


# The workspace is a clone of a third-party repository, and code from it ran
# in the sandbox. Never let its .git/config or hooks run code on the host.
SAFE_GIT_CONFIG = [
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.fsmonitor=false",
]


def _git(workspace: str, args: list[str], token: str = "", check: bool = True):
    result = subprocess.run(
        ["git", *SAFE_GIT_CONFIG, *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT,
        env=_git_env(token) if token else None,
    )

    if check and result.returncode != 0:
        message = _redact(
            (result.stderr or result.stdout).strip(), token
        )

        raise PublishError(f"git {args[0]} failed: {message[-800:]}")

    return result


def _base_branch(workspace: str) -> str:
    """Branch the pull request targets: the one that is checked out."""
    base = _git(workspace, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()

    if base != "HEAD":
        return base

    # Detached HEAD: fall back to the remote's default branch.
    default = _git(
        workspace, ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        check=False,
    ).stdout.strip()

    return default.removeprefix("origin/") or "main"


def _free_branch_name(workspace: str, branch: str) -> str:
    """`branch`, or `branch-2`, `branch-3`... if it already exists locally.

    Publishing the same run twice (e.g. after a failed push) must not die
    on "branch already exists".
    """
    candidate = branch

    for number in range(2, 12):
        exists = _git(
            workspace,
            ["rev-parse", "--verify", "--quiet", f"refs/heads/{candidate}"],
            check=False,
        ).returncode == 0

        if not exists:
            return candidate

        candidate = f"{branch}-{number}"

    raise PublishError(f"Too many existing branches named {branch}.")


def build_pull_request_body(state: dict) -> str:
    """Pull request description assembled from the run, without an LLM."""
    review = state.get("review_report") or {}
    test_report = state.get("test_report") or {}
    files = state.get("applied_files") or []

    sections = [f"**Request**\n\n{state.get('request', '').strip()}"]

    if state.get("implementation_summary"):
        sections.append(
            f"**What changed**\n\n{state['implementation_summary'].strip()}"
        )

    if files:
        sections.append(
            "**Files**\n\n" + "\n".join(f"- `{path}`" for path in files)
        )

    if test_report:
        sections.append(
            "**Checks**\n\n"
            f"{test_report.get('overall_status', 'unknown')}: "
            f"{test_report.get('analysis', '').strip()}"
        )

    if review.get("summary"):
        sections.append(
            f"**AI review** ({review.get('overall_status', '')})\n\n"
            f"{review['summary'].strip()}"
        )

    sections.append(
        "_Generated by CodeCrew and approved by a human reviewer._"
    )

    return "\n\n".join(sections)


def _open_pull_request(
    owner: str, repo: str, token: str,
    title: str, body: str, head: str, base: str,
) -> str:
    request = urllib.request.Request(
        f"{GITHUB_API}/repos/{owner}/{repo}/pulls",
        data=json.dumps({
            "title": title, "head": head, "base": base, "body": body,
        }).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "CodeCrew",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
            return json.loads(response.read())["html_url"]

    except urllib.error.HTTPError as error:
        detail = _redact(error.read().decode(errors="replace"), token)
        raise PublishError(
            f"GitHub refused the pull request ({error.code}): {detail[:500]}"
        )

    except (urllib.error.URLError, OSError) as error:
        raise PublishError(f"Could not reach GitHub: {error}")


def publish_changes(
    workspace: str,
    applied_files: list[str],
    title: str,
    body: str,
    branch: str,
    token: str,
) -> dict:
    """Commit `applied_files` on a new branch, push it, open a PR.

    Only the files agents actually changed are committed, so CodeCrew's own
    working files (virtualenv, caches) never end up in the commit.
    """
    if not token:
        raise PublishError("GITHUB_TOKEN is not set.")

    files = []

    for path in applied_files:
        relative = safe_relative_path(path).as_posix()

        if (Path(workspace) / relative).exists() and relative not in files:
            files.append(relative)

    if not files:
        raise PublishError("There are no changed files to publish.")

    remote = _git(workspace, ["remote", "get-url", "origin"]).stdout
    owner, repo = parse_github_repo(remote)

    base = _base_branch(workspace)
    branch = _free_branch_name(workspace, branch)

    _git(workspace, ["checkout", "-b", branch])
    _git(workspace, ["add", "--", *files])

    nothing_staged = _git(
        workspace, ["diff", "--cached", "--quiet"], check=False
    ).returncode == 0

    if nothing_staged:
        raise PublishError("The agents' changes are identical to the repository.")

    _git(workspace, [
        "-c", f"user.name={os.getenv('CODECREW_GIT_NAME', DEFAULT_GIT_NAME)}",
        "-c", f"user.email={os.getenv('CODECREW_GIT_EMAIL', DEFAULT_GIT_EMAIL)}",
        "commit", "-m", title, "-m", body,
    ])

    try:
        _git(workspace, ["push", "-u", "origin", branch], token=token)
    except PublishError as error:
        raise PublishError(
            f"{error} (If you do not have push access to {owner}/{repo}, "
            "fork it and run CodeCrew on your fork.)"
        )

    pr_url = _open_pull_request(owner, repo, token, title, body, branch, base)

    return {"branch": branch, "pr_url": pr_url, "base": base, "files": files}
