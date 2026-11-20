"""Pick the repository files that matter for a task and format them.

A model sees only a few thousand characters of code per request, so
sending "the first files we found" wastes most of that. Instead, files the
architect named come first, then the files that best match the task text.
"""
import os
import re
from pathlib import Path

from backend.app.utils.ignore import (
    IGNORED_DIRS,
    IGNORED_FILES,
    IGNORED_SUFFIXES,
    is_sensitive_file,
)
from backend.app.utils.paths import resolve_in_workspace

MAX_SCANNED_FILE_BYTES = 60_000
MAX_SCANNED_FILES = 2_000

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "must",
    "should", "will", "when", "then", "than", "have", "has", "not", "are",
    "any", "all", "can", "its", "use", "using", "used", "new", "add",
    "returns", "return", "file", "files", "task", "code", "each", "also",
    "only", "such", "which", "their", "there", "else", "none", "true",
    "false",
}


def tokenize(text: str) -> set[str]:
    """Lower-case identifier words, splitting snake_case and camelCase."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    words = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", text.replace("_", " "))

    return {
        word.lower() for word in words
        if word.lower() not in STOPWORDS
    }


def _candidate_files(root: Path):
    scanned = 0

    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            name for name in dirs
            if name not in IGNORED_DIRS
            and not (Path(current) / name).is_symlink()
        )

        for name in sorted(files):
            path = Path(current) / name

            if (
                name.lower() in IGNORED_FILES
                or is_sensitive_file(name)
                or path.suffix.lower() in IGNORED_SUFFIXES
                or path.is_symlink()
            ):
                continue

            try:
                if path.stat().st_size > MAX_SCANNED_FILE_BYTES:
                    continue
            except OSError:
                continue

            scanned += 1

            if scanned > MAX_SCANNED_FILES:
                return

            yield path


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None

    if b"\x00" in data[:2048]:
        return None  # binary

    return data.decode("utf-8", errors="replace")


def _score(rel_path: str, text: str, query: set[str], near_dirs: set[str]) -> int:
    path_tokens = tokenize(rel_path)
    lowered = text.lower()

    score = 3 * len(query & path_tokens)
    score += sum(1 for token in query if token in lowered)

    parent = os.path.dirname(rel_path)

    if parent in near_dirs:
        score += 2

    return score


def find_relevant_files(
    workspace: str,
    query: str,
    explicit: list[str] | None = None,
    limit: int = 6,
    min_score: int = 3,
) -> list[str]:
    """Repository-relative paths worth showing for `query`.

    `explicit` (paths the architect named) come first when they exist; the
    rest are the best keyword matches, boosted when they sit next to an
    explicit file, since neighbours show the local conventions.
    """
    root = Path(workspace).resolve()
    explicit = list(dict.fromkeys(explicit or []))

    selected = []

    for path in explicit:
        try:
            full = resolve_in_workspace(workspace, path)
        except ValueError:
            continue

        if full.is_file():
            selected.append(full.relative_to(root).as_posix())

    explicit_count = len(selected)
    near_dirs = {os.path.dirname(path) for path in explicit}
    query_tokens = tokenize(query)

    scored = []

    for path in _candidate_files(root):
        rel_path = path.relative_to(root).as_posix()

        if rel_path in selected:
            continue

        text = _read_text(path)

        if text is None:
            continue

        score = _score(rel_path, text, query_tokens, near_dirs)

        if score >= min_score:
            scored.append((-score, rel_path))

    scored.sort()

    for _, rel_path in scored:
        if len(selected) >= limit + explicit_count:
            break

        selected.append(rel_path)

    return selected


def read_files_text(
    workspace: str,
    rel_paths: list[str],
    char_budget: int,
    min_chunk: int = 300,
) -> str:
    """Format files as `FILE: path` blocks within a character budget.

    Earlier files get the space first; a file that does not fit is cut and
    marked, and the remaining ones are listed as omitted.
    """
    blocks = []
    remaining = max(0, char_budget)
    omitted = []

    for rel_path in rel_paths:
        try:
            full = resolve_in_workspace(workspace, rel_path)
        except ValueError:
            continue

        text = _read_text(full) if full.is_file() else None

        if text is None:
            continue

        header = f"FILE: {rel_path}\n"
        room = remaining - len(header)

        if room < min_chunk:
            omitted.append(rel_path)
            continue

        if len(text) > room:
            text = text[:room] + "\n... (file truncated)"

        blocks.append(header + text)
        remaining -= len(header) + len(text) + 2

    if omitted:
        blocks.append("(not shown, no room: " + ", ".join(omitted) + ")")

    return "\n\n".join(blocks)
