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

    # A second bare repository stands in for the token owner's fork.
    fork = tmp_path / "fork.git"
    fork.mkdir()
    git(fork, "init", "--bare", "-b", "main")

    pulls = []
    pull_options = []
    access = {"can_push": True, "allow_forking": True}

    monkeypatch.delenv("CODECREW_PUBLISH_MODE", raising=False)
    monkeypatch.setattr(publisher.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        publisher, "parse_github_repo", lambda url: ("owner", "repo")
    )
    monkeypatch.setattr(publisher, "_repo_access", lambda *args: dict(access))
    monkeypatch.setattr(
        publisher, "_ensure_fork", lambda *args: ("forker", "repo-fork")
    )
    monkeypatch.setattr(
        publisher, "_fork_remote_url", lambda *args: str(fork)
    )

    def open_pull_request(*args, **kwargs):
        pulls.append(args)
        pull_options.append(kwargs)
        return "https://github.com/owner/repo/pull/7"

    monkeypatch.setattr(publisher, "_open_pull_request", open_pull_request)

    return {
        "origin": origin, "fork": fork, "work": work, "pulls": pulls,
        "pull_options": pull_options, "access": access,
    }


def publish(repo, files, **kwargs):
    return publish_changes(
        str(repo["work"]), files,
        kwargs.get("title", "CodeCrew: add health"),
        kwargs.get("body", "details"),
        kwargs.get("branch", "codecrew/add-health-abc12345"),
        kwargs.get("token", "t0ken"),
        mode=kwargs.get("mode"),
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

    assert "CODECREW_PUBLISH_MODE=fork" in str(error.value)
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


# ---------------------------------------------------------- direct vs fork

def branches(bare):
    return git(bare, "branch", "--format=%(refname:short)").split()


def test_write_access_pushes_to_the_repository_itself(repo):
    (repo["work"] / "a.py").write_text("x\n")

    result = publish(repo, ["a.py"])

    assert result["mode"] == "direct"
    assert result["head"] == result["branch"]
    assert result["target"] == "owner/repo"
    assert result["branch"] in branches(repo["origin"])
    assert branches(repo["fork"]) == []
    assert repo["pull_options"][0] == {"maintainer_can_modify": False}


def test_without_write_access_the_branch_goes_to_a_fork(repo):
    repo["access"]["can_push"] = False
    (repo["work"] / "a.py").write_text("x\n")

    result = publish(repo, ["a.py"])
    branch = result["branch"]

    assert result["mode"] == "fork"
    assert branch in branches(repo["fork"])
    assert branch not in branches(repo["origin"])          # upstream untouched
    assert git(repo["fork"], "ls-tree", "-r", "--name-only", branch).split() == [
        "README.md", "a.py"
    ]

    # The pull request is opened on the ORIGINAL repository, from the fork.
    owner, repo_name, _, _, _, head, base = repo["pulls"][0]
    assert (owner, repo_name) == ("owner", "repo")
    assert head == f"forker:{branch}"
    assert base == "main"
    assert repo["pull_options"][0] == {"maintainer_can_modify": True}


def test_fork_can_be_forced_even_with_write_access(repo):
    (repo["work"] / "a.py").write_text("x\n")

    result = publish(repo, ["a.py"], mode="fork")

    assert result["mode"] == "fork"
    assert branches(repo["origin"]) == ["main"]


def test_direct_can_be_forced_without_write_access(repo):
    repo["access"]["can_push"] = False
    (repo["work"] / "a.py").write_text("x\n")

    result = publish(repo, ["a.py"], mode="direct")

    assert result["mode"] == "direct"
    assert result["branch"] in branches(repo["origin"])


def test_mode_is_read_from_the_environment(repo, monkeypatch):
    monkeypatch.setenv("CODECREW_PUBLISH_MODE", "FORK")
    (repo["work"] / "a.py").write_text("x\n")

    assert publish(repo, ["a.py"])["mode"] == "fork"


def test_unknown_mode_is_rejected(repo):
    with pytest.raises(PublishError, match="CODECREW_PUBLISH_MODE"):
        publish(repo, ["README.md"], mode="yolo")


def test_repository_that_forbids_forking_fails_before_committing(repo):
    repo["access"].update(can_push=False, allow_forking=False)
    (repo["work"] / "a.py").write_text("x\n")

    with pytest.raises(PublishError, match="does not allow forking"):
        publish(repo, ["a.py"])

    assert "codecrew" not in git(repo["work"], "branch")   # nothing was created


def test_a_failing_access_check_stops_before_any_local_change(repo, monkeypatch):
    def refuse(*args):
        raise PublishError("not found")

    monkeypatch.setattr(publisher, "_repo_access", refuse)
    (repo["work"] / "a.py").write_text("x\n")

    with pytest.raises(PublishError, match="not found"):
        publish(repo, ["a.py"])

    assert git(repo["work"], "branch", "--list").split() == ["*", "main"]


def test_fork_push_is_retried_while_the_fork_warms_up(monkeypatch):
    calls = []

    def fake_git(workspace, args, token="", check=True):
        calls.append(args)

        if args[:2] == ["remote", "get-url"]:
            return type("R", (), {"returncode": 1})()

        if args[0] == "push" and len([c for c in calls if c[0] == "push"]) < 3:
            raise PublishError("git push failed: repository not ready")

        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(publisher, "_git", fake_git)
    monkeypatch.setattr(publisher.time, "sleep", lambda seconds: None)

    publisher._push_to_fork("/w", "https://x/fork.git", "br", "tok")

    assert [c for c in calls if c[0] == "push"] == [
        ["push", "-u", "codecrew-fork", "br"]
    ] * 3
    assert ["remote", "add", "codecrew-fork", "https://x/fork.git"] in calls


def test_fork_push_gives_up_after_the_attempt_limit(monkeypatch):
    def fake_git(workspace, args, token="", check=True):
        if args[:2] == ["remote", "get-url"]:
            return type("R", (), {"returncode": 0})()

        if args[0] == "push":
            raise PublishError("git push failed: denied")

        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(publisher, "_git", fake_git)
    monkeypatch.setattr(publisher.time, "sleep", lambda seconds: None)

    with pytest.raises(PublishError, match="denied"):
        publisher._push_to_fork("/w", "u", "br", "tok")


# ------------------------------------------------------------ GitHub calls

class FakeGitHub:
    """Replaces publisher._github_api with scripted (status, json) replies."""

    def __init__(self, monkeypatch, replies):
        self.replies = list(replies)
        self.calls = []
        monkeypatch.setattr(publisher, "_github_api", self)
        monkeypatch.setattr(publisher.time, "sleep", lambda seconds: None)

    def __call__(self, token, method, path, payload=None):
        self.calls.append((method, path, payload))

        if not self.replies:
            raise AssertionError(f"unexpected call {method} {path}")

        return self.replies.pop(0)


def test_repo_access_reads_the_push_permission(monkeypatch):
    FakeGitHub(monkeypatch, [(200, {"permissions": {"push": False}})])

    assert publisher._repo_access("o", "r", "t") == {
        "can_push": False, "allow_forking": True
    }


def test_repo_access_not_found_mentions_private_repositories(monkeypatch):
    FakeGitHub(monkeypatch, [(404, {"message": "Not Found"})])

    with pytest.raises(PublishError, match="not found.*private"):
        publisher._repo_access("o", "r", "t")


def test_repo_access_rejects_archived_repositories(monkeypatch):
    FakeGitHub(monkeypatch, [(200, {"archived": True})])

    with pytest.raises(PublishError, match="archived"):
        publisher._repo_access("o", "r", "t")


def test_repo_access_reports_other_errors_without_the_token(monkeypatch):
    FakeGitHub(monkeypatch, [(401, {"message": "Bad credentials SECRET-T"})])

    with pytest.raises(PublishError) as error:
        publisher._repo_access("o", "r", "SECRET-T")

    assert "401" in str(error.value) and "SECRET-T" not in str(error.value)


def test_ensure_fork_waits_until_the_fork_exists(monkeypatch):
    github = FakeGitHub(monkeypatch, [
        (202, {"owner": {"login": "forker"}, "name": "repo"}),
        (404, {}), (404, {}), (200, {}),
    ])

    assert publisher._ensure_fork("owner", "repo", "t") == ("forker", "repo")
    assert github.calls[0][:2] == ("POST", "/repos/owner/repo/forks")
    assert [c[0] for c in github.calls] == ["POST", "GET", "GET", "GET"]
    assert github.calls[1][1] == "/repos/forker/repo"


def test_ensure_fork_uses_the_name_github_gave_the_fork(monkeypatch):
    FakeGitHub(monkeypatch, [
        (202, {"owner": {"login": "forker"}, "name": "repo-1"}), (200, {}),
    ])

    assert publisher._ensure_fork("owner", "repo", "t") == ("forker", "repo-1")


def test_ensure_fork_explains_missing_permission(monkeypatch):
    FakeGitHub(monkeypatch, [(403, {"message": "Resource not accessible"})])

    with pytest.raises(PublishError, match="public_repo"):
        publisher._ensure_fork("owner", "repo", "t")


def test_ensure_fork_refuses_to_fork_your_own_repository(monkeypatch):
    FakeGitHub(monkeypatch, [(202, {"owner": {"login": "Owner"}, "name": "repo"})])

    with pytest.raises(PublishError, match="CODECREW_PUBLISH_MODE=direct"):
        publisher._ensure_fork("owner", "repo", "t")


def test_ensure_fork_gives_up_if_the_fork_never_appears(monkeypatch):
    monkeypatch.setattr(publisher, "FORK_POLL_ATTEMPTS", 3)
    github = FakeGitHub(monkeypatch, [
        (202, {"owner": {"login": "forker"}, "name": "repo"}),
        (404, {}), (404, {}), (404, {}),
    ])

    with pytest.raises(PublishError, match="not ready"):
        publisher._ensure_fork("owner", "repo", "t")

    assert len(github.calls) == 4


def test_pull_request_request_has_the_fork_head_and_maintainer_flag(monkeypatch):
    github = FakeGitHub(monkeypatch, [(201, {"html_url": "https://github.com/o/r/pull/1"})])

    url = publisher._open_pull_request(
        "o", "r", "t", "Title", "Body", "forker:br", "main",
        maintainer_can_modify=True,
    )

    method, path, payload = github.calls[0]
    assert (method, path) == ("POST", "/repos/o/r/pulls")
    assert payload == {
        "title": "Title", "head": "forker:br", "base": "main",
        "body": "Body", "maintainer_can_modify": True,
    }
    assert url.endswith("/pull/1")


def test_refused_pull_request_shows_githubs_reason(monkeypatch):
    FakeGitHub(monkeypatch, [(422, {
        "message": "Validation Failed",
        "errors": [{"message": "A pull request already exists"}],
    })])

    with pytest.raises(PublishError, match="422.*already exists"):
        publisher._open_pull_request(
            "o", "r", "ghp_TOKEN123", "T", "B", "br", "main"
        )
