"""Publish approved changes: branch, commit, push and open a pull request.

Runs only after a human approved the result. Needs a GitHub token
(GITHUB_TOKEN). Where the branch is pushed depends on what the token may do
with the project's repository:

* write access (your own repository, or you are a collaborator):
  the branch is pushed to the repository itself;
* no write access (someone else's project): the repository is forked into
  the token's account, the branch is pushed to the fork, and a pull request
  is opened from the fork against the original. The owner can review it and
  merge it if they want to.

CODECREW_PUBLISH_MODE can force `direct` or `fork`; the default is `auto`.

When the token is missing or publishing fails, the changes stay in the
workspace and the reason is reported; nothing here raises into the workflow.

The token is passed to git through environment variables, never on the
command line, so it does not show up in process lists or logs.
"""
import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from backend.app.utils.paths import safe_relative_path

GIT_TIMEOUT = 120
API_TIMEOUT = 30
GITHUB_API = "https://api.github.com"

# A fork is created asynchronously by GitHub; wait for it to become usable.
FORK_POLL_ATTEMPTS = 20
FORK_POLL_SECONDS = 3
FORK_PUSH_ATTEMPTS = 3
FORK_REMOTE = "codecrew-fork"

PUBLISH_MODES = ("auto", "direct", "fork")

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


def _github_api(token: str, method: str, path: str, payload=None):
    """Call the GitHub REST API. Returns (status code, parsed JSON).

    HTTP error statuses are returned, not raised, so callers can react to
    404/403/422. Only network problems raise.
    """
    request = urllib.request.Request(
        f"{GITHUB_API}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "CodeCrew",
        },
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if raw else {})

    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")

        try:
            data = json.loads(raw)
        except ValueError:
            data = {"message": raw}

        return error.code, data if isinstance(data, dict) else {}

    except (urllib.error.URLError, OSError) as error:
        raise PublishError(f"Could not reach GitHub: {error}")


def _api_message(data: dict, token: str) -> str:
    message = data.get("message", "")
    details = "; ".join(
        str(item.get("message", item)) if isinstance(item, dict) else str(item)
        for item in data.get("errors", []) or []
    )

    return _redact(f"{message} {details}".strip(), token)[:500]


def _repo_access(owner: str, repo: str, token: str) -> dict:
    """What the token may do with the repository."""
    status, data = _github_api(token, "GET", f"/repos/{owner}/{repo}")

    if status == 404:
        raise PublishError(
            f"Repository {owner}/{repo} was not found, or the token cannot "
            "see it (a private repository needs a token with access to it)."
        )

    if status != 200:
        raise PublishError(
            f"Could not read {owner}/{repo} ({status}): "
            f"{_api_message(data, token)}"
        )

    if data.get("archived"):
        raise PublishError(f"{owner}/{repo} is archived and read-only.")

    return {
        "can_push": bool((data.get("permissions") or {}).get("push")),
        "allow_forking": data.get("allow_forking", True),
    }


def _ensure_fork(owner: str, repo: str, token: str) -> tuple[str, str]:
    """Fork owner/repo into the token's account; return (owner, name).

    If the account already has a fork, GitHub returns it. Forking is
    asynchronous, so wait until the fork can be read.
    """
    status, data = _github_api(
        token, "POST", f"/repos/{owner}/{repo}/forks", {}
    )

    if status not in (200, 201, 202):
        raise PublishError(
            f"Could not fork {owner}/{repo} ({status}): "
            f"{_api_message(data, token)} The token must be allowed to "
            "create forks (a classic token with the `public_repo` scope, "
            "or `repo` for private repositories)."
        )

    fork_owner = data["owner"]["login"]
    fork_repo = data["name"]

    if fork_owner.lower() == owner.lower():
        raise PublishError(
            f"The token belongs to the owner of {owner}/{repo}, so it cannot "
            "be forked. Use CODECREW_PUBLISH_MODE=direct."
        )

    for _ in range(FORK_POLL_ATTEMPTS):
        ready, _ = _github_api(
            token, "GET", f"/repos/{fork_owner}/{fork_repo}"
        )

        if ready == 200:
            return fork_owner, fork_repo

        time.sleep(FORK_POLL_SECONDS)

    raise PublishError(
        f"The fork {fork_owner}/{fork_repo} was created but is not ready "
        "yet. Publish again in a minute."
    )


def _fork_remote_url(fork_owner: str, fork_repo: str) -> str:
    return f"https://github.com/{fork_owner}/{fork_repo}.git"


def _push_to_fork(workspace: str, url: str, branch: str, token: str) -> None:
    """Push to the fork, retrying: a fresh fork may not accept pushes at once."""
    has_remote = _git(
        workspace, ["remote", "get-url", FORK_REMOTE], check=False
    ).returncode == 0

    _git(workspace, [
        "remote", "set-url" if has_remote else "add", FORK_REMOTE, url
    ])

    last_error = None

    for attempt in range(FORK_PUSH_ATTEMPTS):
        try:
            _git(
                workspace, ["push", "-u", FORK_REMOTE, branch], token=token
            )
            return
        except PublishError as error:
            last_error = error

            if attempt < FORK_PUSH_ATTEMPTS - 1:
                time.sleep(FORK_POLL_SECONDS)

    raise last_error


def _open_pull_request(
    owner: str, repo: str, token: str,
    title: str, body: str, head: str, base: str,
    maintainer_can_modify: bool = False,
) -> str:
    """Open the PR on owner/repo. `head` is `branch` or `fork_owner:branch`."""
    status, data = _github_api(
        token, "POST", f"/repos/{owner}/{repo}/pulls", {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
            "maintainer_can_modify": maintainer_can_modify,
        },
    )

    if status not in (200, 201):
        raise PublishError(
            f"GitHub refused the pull request ({status}): "
            f"{_api_message(data, token)}"
        )

    return data["html_url"]


def publish_changes(
    workspace: str,
    applied_files: list[str],
    title: str,
    body: str,
    branch: str,
    token: str,
    mode: str | None = None,
) -> dict:
    """Commit `applied_files` on a new branch, push it, open a PR.

    The PR always targets the repository the workspace was cloned from.
    `mode` is "direct" (push to that repository), "fork" (push to a fork
    and open a cross-repository PR) or "auto" (direct when the token has
    write access, otherwise fork). Defaults to CODECREW_PUBLISH_MODE, then
    "auto".

    Only the files agents actually changed are committed, so CodeCrew's own
    working files (virtualenv, caches) never end up in the commit.
    """
    if not token:
        raise PublishError("GITHUB_TOKEN is not set.")

    mode = (mode or os.getenv("CODECREW_PUBLISH_MODE") or "auto").lower()

    if mode not in PUBLISH_MODES:
        raise PublishError(
            f"CODECREW_PUBLISH_MODE must be one of {PUBLISH_MODES}, "
            f"not {mode!r}."
        )

    files = []

    for path in applied_files:
        relative = safe_relative_path(path).as_posix()

        if (Path(workspace) / relative).exists() and relative not in files:
            files.append(relative)

    if not files:
        raise PublishError("There are no changed files to publish.")

    remote = _git(workspace, ["remote", "get-url", "origin"]).stdout
    owner, repo = parse_github_repo(remote)

    # Ask GitHub first: it is cheap and fails fast on a wrong repository or
    # token, before anything is committed.
    access = _repo_access(owner, repo, token)
    use_fork = mode == "fork" or (mode == "auto" and not access["can_push"])

    if use_fork and not access["allow_forking"]:
        raise PublishError(
            f"{owner}/{repo} does not allow forking, and the token has no "
            "write access to it, so a pull request cannot be opened."
        )

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

    if use_fork:
        fork_owner, fork_repo = _ensure_fork(owner, repo, token)

        _push_to_fork(
            workspace, _fork_remote_url(fork_owner, fork_repo), branch, token
        )

        head = f"{fork_owner}:{branch}"

    else:
        try:
            _git(workspace, ["push", "-u", "origin", branch], token=token)
        except PublishError as error:
            raise PublishError(
                f"{error} (Check that the token may write to {owner}/{repo}. "
                "Set CODECREW_PUBLISH_MODE=fork to publish through a fork "
                "instead.)"
            )

        head = branch

    pr_url = _open_pull_request(
        owner, repo, token, title, body, head, base,
        maintainer_can_modify=use_fork,
    )

    return {
        "branch": branch,
        "pr_url": pr_url,
        "base": base,
        "files": files,
        "mode": "fork" if use_fork else "direct",
        "head": head,
        "target": f"{owner}/{repo}",
    }
