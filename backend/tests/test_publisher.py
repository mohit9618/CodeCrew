"""Publishing: real git against a local bare repository; GitHub is faked."""
import subprocess

import pytest

from backend.app.services import publisher
from backend.app.services.publisher import (
    PublishError,
    build_branch_name,
    build_pull_request_body,
    parse_github_repo,
    publish_changes,
    slugify,
)

IDENTITY = ["-c", "user.name=t", "-c", "user.email=t@example.com"]


def git(cwd, *args):
    return subprocess.run(
        ["git", *IDENTITY, *args], cwd=cwd, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A clone of a bare 'origin' with one commit on main."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    origin.mkdir()
    work.mkdir()

    git(origin, "init", "--bare", "-b", "main")
    git(work, "init", "-b", "main")
    (work / "README.md").write_text("hello\n")
    git(work, "add", "README.md")
    git(work, "commit", "-m", "initial")
    git(work, "remote", "add", "origin", str(origin))
    git(work, "push", "-u", "origin", "main")

    pulls = []
    monkeypatch.setattr(
        publisher, "parse_github_repo", lambda url: ("owner", "repo")
    )
    monkeypatch.setattr(
        publisher, "_open_pull_request",
        lambda *args: pulls.append(args) or "https://github.com/owner/repo/pull/7",
    )

    return {"origin": origin, "work": work, "pulls": pulls}


def publish(repo, files, **kwargs):
    return publish_changes(
        str(repo["work"]), files,
        kwargs.get("title", "CodeCrew: add health"),
        kwargs.get("body", "details"),
        kwargs.get("branch", "codecrew/add-health-abc12345"),
        kwargs.get("token", "t0ken"),
    )


def test_commits_only_the_applied_files_and_opens_a_pull_request(repo):
    work = repo["work"]
    (work / "app").mkdir()
    (work / "app" / "health.py").write_text("x = 1\n")
    (work / ".codecrew").mkdir()
    (work / ".codecrew" / "install.json").write_text("{}")   # CodeCrew's own
    (work / "scratch.txt").write_text("not part of the change")

    result = publish(repo, ["app/health.py"])

    assert result["pr_url"].endswith("/pull/7")
    assert result["base"] == "main"

    branch = result["branch"]
    files = git(repo["origin"], "ls-tree", "-r", "--name-only", branch).split()
    assert files == ["README.md", "app/health.py"]

    assert git(repo["origin"], "log", "-1", "--format=%s", branch) == "CodeCrew: add health"

    # The pull request targets the right branches.
    _, _, _, title, body, head, base = repo["pulls"][0]
    assert (head, base) == (branch, "main")


def test_missing_and_duplicate_files_are_ignored(repo):
    (repo["work"] / "a.py").write_text("x\n")

    result = publish(repo, ["a.py", "a.py", "does/not/exist.py"])

    assert result["files"] == ["a.py"]


def test_modified_tracked_file_is_committed(repo):
    (repo["work"] / "README.md").write_text("changed\n")

    publish(repo, ["README.md"])

    shown = git(repo["origin"], "show", "codecrew/add-health-abc12345:README.md")
    assert shown == "changed"


def test_no_changes_is_reported_not_committed(repo):
    with pytest.raises(PublishError, match="identical"):
        publish(repo, ["README.md"])


def test_no_files_to_publish(repo):
    with pytest.raises(PublishError, match="no changed files"):
        publish(repo, [])


def test_missing_token_is_reported(repo):
    (repo["work"] / "a.py").write_text("x\n")

    with pytest.raises(PublishError, match="GITHUB_TOKEN"):
        publish(repo, ["a.py"], token="")


@pytest.mark.parametrize("path", ["../x.py", ".env", ".git/config"])
def test_unsafe_paths_are_refused(repo, path):
    with pytest.raises(ValueError):
        publish(repo, [path])


def test_refused_push_explains_forking_and_hides_the_token(repo, monkeypatch):
    (repo["work"] / "a.py").write_text("x\n")
    git(repo["work"], "remote", "set-url", "origin", "/nonexistent/path.git")

    with pytest.raises(PublishError) as error:
        publish(repo, ["a.py"], token="SECRET-TOKEN")

    assert "fork it" in str(error.value)
    assert "SECRET-TOKEN" not in str(error.value)
    assert repo["pulls"] == []   # no pull request without a push


def test_token_is_passed_through_the_environment_not_argv():
    env = publisher._git_env("SECRET-TOKEN")

    assert "SECRET-TOKEN" not in " ".join(
        value for key, value in env.items() if key.startswith("GIT_CONFIG_KEY")
    )
    assert env["GIT_CONFIG_VALUE_0"].startswith("AUTHORIZATION: basic ")
    assert env["GIT_TERMINAL_PROMPT"] == "0"


# ------------------------------------------------------------ small helpers

@pytest.mark.parametrize("url, expected", [
    ("https://github.com/pallets/flask", ("pallets", "flask")),
    ("https://github.com/pallets/flask.git\n", ("pallets", "flask")),
    ("git@github.com:me/my.repo.git", ("me", "my.repo")),
    ("https://github.com/me/repo/", ("me", "repo")),
])
def test_parse_github_repo(url, expected):
    assert parse_github_repo(url) == expected


def test_parse_github_repo_rejects_other_hosts():
    with pytest.raises(PublishError):
        parse_github_repo("https://gitlab.com/me/repo")


def test_branch_names_are_safe_and_unique_per_run():
    first = build_branch_name("Add /health endpoint!! (JSON)", "/w/aaaa1111-2222")
    second = build_branch_name("Add /health endpoint!! (JSON)", "/w/bbbb1111-2222")

    assert first == "codecrew/add-health-endpoint-json-aaaa1111"
    assert first != second
    assert slugify("???") == "change"
    assert len(slugify("x" * 200)) <= 40


def test_pull_request_body_summarises_the_run():
    body = build_pull_request_body({
        "request": "Add a health endpoint",
        "implementation_summary": "Added /health.",
        "applied_files": ["app/health.py"],
        "test_report": {"overall_status": "passed", "analysis": "All green."},
        "review_report": {"overall_status": "approved", "summary": "Looks good."},
    })

    for expected in ("Add a health endpoint", "Added /health.",
                     "`app/health.py`", "All green.", "Looks good.",
                     "approved by a human"):
        assert expected in body


def test_pull_request_body_works_with_a_minimal_state():
    assert "approved by a human" in build_pull_request_body({"request": "x"})
