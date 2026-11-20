"""Path safety, search/replace edits, file retrieval and prompt budgets."""
from types import SimpleNamespace

import pytest

from backend.app.utils.edits import EditError, apply_edits
from backend.app.utils.ignore import is_sensitive_file
from backend.app.utils.paths import resolve_in_workspace, safe_relative_path
from backend.app.utils.retrieval import (
    find_relevant_files,
    read_files_text,
    tokenize,
)
from backend.app.utils.text import split_budget


def edit(search, replace):
    return SimpleNamespace(search=search, replace=replace)


def write(root, relative, text=""):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# -------------------------------------------------------------------- paths

@pytest.mark.parametrize("path", [
    "", "   ", "/etc/passwd", "C:/Windows/x.py", "C:\\x.py",
    "../outside.py", "a/../../outside.py",
    ".git/config", ".codecrew/venv/bin/python", "node_modules/x/index.js",
    ".env", "config/.env.production", "keys/server.pem", "id_rsa",
])
def test_unsafe_paths_are_rejected(path):
    with pytest.raises(ValueError):
        safe_relative_path(path)


def test_normal_paths_are_accepted_and_normalised():
    assert safe_relative_path("app\\api\\routes.py").as_posix() == "app/api/routes.py"
    assert safe_relative_path("src/.github/ci.yml").as_posix() == "src/.github/ci.yml"


def test_resolve_stays_inside_workspace(tmp_path):
    resolved = resolve_in_workspace(str(tmp_path), "pkg/mod.py")

    assert resolved == (tmp_path / "pkg" / "mod.py").resolve()


def test_lock_files_are_ignored_but_not_secret():
    assert is_sensitive_file(".env")
    assert is_sensitive_file("cert.pem")
    assert not is_sensitive_file("package-lock.json")


# -------------------------------------------------------------------- edits

def test_edit_replaces_text_once():
    result = apply_edits("a = 1\nb = 2\n", [edit("b = 2", "b = 3")])

    assert result == "a = 1\nb = 3\n"


def test_edits_apply_in_order():
    result = apply_edits("x", [edit("x", "y"), edit("y", "z")])

    assert result == "z"


def test_edit_search_not_found_names_the_closest_line():
    with pytest.raises(EditError) as error:
        apply_edits("def add(a, b):\n    return a + b\n",
                    [edit("def add(x, y):", "def add(x, y, z):")], "calc.py")

    message = str(error.value)
    assert "not found" in message
    assert "def add(a, b):" in message  # tells the model what exists


def test_edit_that_matches_twice_is_rejected():
    with pytest.raises(EditError, match="2 times"):
        apply_edits("x = 1\nx = 1\n", [edit("x = 1", "x = 2")])


def test_empty_search_is_rejected():
    with pytest.raises(EditError, match="empty search"):
        apply_edits("abc", [edit("  ", "x")])


def test_no_edits_is_rejected():
    with pytest.raises(EditError):
        apply_edits("abc", [])


def test_crlf_files_can_be_edited_with_lf_snippets_and_stay_crlf():
    result = apply_edits(
        "one\r\ntwo\r\nthree\r\n", [edit("two\nthree", "TWO\nTHREE")]
    )

    assert result == "one\r\nTWO\r\nTHREE\r\n"


# ---------------------------------------------------------------- retrieval

def test_tokenize_splits_identifiers_and_drops_filler():
    tokens = tokenize("Add getUserProfile endpoint for the user_profile table")

    assert {"get", "user", "profile", "endpoint", "table"} <= tokens
    assert "the" not in tokens and "for" not in tokens


def test_explicit_files_come_first_then_best_matches(tmp_path):
    write(tmp_path, "app/health.py", "def health():\n    return 'ok'\n")
    write(tmp_path, "app/users.py", "def list_users():\n    return []\n")
    write(tmp_path, "docs/notes.md", "nothing relevant here")
    write(tmp_path, "tests/test_users.py", "from app.users import list_users\n")

    found = find_relevant_files(
        str(tmp_path),
        "Add a users endpoint that lists users",
        explicit=["app/health.py", "app/missing.py"],
    )

    assert found[0] == "app/health.py"            # explicit and exists
    assert "app/missing.py" not in found           # does not exist
    assert "app/users.py" in found                 # matches the task
    assert "docs/notes.md" not in found            # unrelated


def test_retrieval_skips_ignored_secret_and_binary_files(tmp_path):
    write(tmp_path, "node_modules/lib/users.js", "users users users")
    write(tmp_path, ".env", "USERS_TOKEN=secret")
    write(tmp_path, "keys/users.pem", "users")
    (tmp_path / "users.bin").write_bytes(b"users\x00\x01\x02")
    write(tmp_path, "src/users.py", "users")

    found = find_relevant_files(str(tmp_path), "users", explicit=[])

    assert found == ["src/users.py"]


def test_retrieval_refuses_explicit_paths_outside_the_workspace(tmp_path):
    write(tmp_path / "repo", "a.py", "x")
    (tmp_path / "secret.txt").write_text("top secret")

    found = find_relevant_files(
        str(tmp_path / "repo"), "x", explicit=["../secret.txt", ".env"]
    )

    assert "../secret.txt" not in found and ".env" not in found


def test_read_files_text_respects_budget_and_lists_omitted(tmp_path):
    write(tmp_path, "a.py", "A" * 1000)
    write(tmp_path, "b.py", "B" * 1000)
    write(tmp_path, "c.py", "C" * 1000)

    text = read_files_text(str(tmp_path), ["a.py", "b.py", "c.py"], 1500)

    assert text.startswith("FILE: a.py")
    assert "(file truncated)" in text
    assert "not shown" in text and "c.py" in text
    assert len(text) < 1500 + 200


def test_read_files_text_skips_missing_and_unsafe_paths(tmp_path):
    write(tmp_path, "ok.py", "print(1)")

    text = read_files_text(
        str(tmp_path), ["missing.py", "../x.py", ".env", "ok.py"], 5000
    )

    assert text == "FILE: ok.py\nprint(1)"


# ------------------------------------------------------------------ budgets

def test_split_budget_never_exceeds_available():
    wanted = {"task": 10_000, "tree": 10_000, "files": 10_000}
    weights = {"task": 0.2, "tree": 0.2, "files": 0.6}

    limits = split_budget(5_000, wanted, weights)

    assert sum(limits.values()) <= 5_000
    assert limits["files"] > limits["task"]


def test_split_budget_gives_unused_space_to_hungry_parts():
    weights = {"task": 0.2, "tree": 0.2, "files": 0.6}

    limits = split_budget(
        10_000, {"task": 100, "tree": 100, "files": 50_000}, weights
    )

    assert limits["task"] == 100 and limits["tree"] == 100
    assert limits["files"] == 9_800  # everything that was left over


def test_split_budget_short_parts_are_not_inflated():
    weights = {"a": 0.5, "b": 0.5}

    assert split_budget(1_000, {"a": 10, "b": 20}, weights) == {"a": 10, "b": 20}
