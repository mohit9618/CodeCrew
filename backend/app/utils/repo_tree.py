import os
from pathlib import Path

from backend.app.utils.ignore import (
    IGNORED_DIRS,
    IGNORED_FILES,
    IGNORED_SUFFIXES,
)

# Hard ceiling for a tree that is embedded in an LLM prompt.
DEFAULT_MAX_TREE_CHARS = 6_000


def build_repo_tree(
    workspace: str,
    max_chars: int = DEFAULT_MAX_TREE_CHARS
) -> str:
    """Return a compact, filtered directory tree for LLM prompts.

    Uses the same ignore rules and the same indented text format as
    RepositoryAnalyzer.inspect_repository, so the tree stays small and
    consistent between the first and later developer tasks. This replaces
    the MCP `directory_tree` tool, which returns verbose, unfiltered JSON
    (including .git, node_modules, venvs, ...).
    """
    root = Path(workspace).resolve()

    if not root.is_dir():
        raise ValueError("Repository workspace does not exist.")

    lines = []
    total_chars = 0
    truncated = False

    for current_dir, dirs, filenames in os.walk(root, followlinks=False):
        current_path = Path(current_dir)

        dirs[:] = sorted(
            name for name in dirs
            if name not in IGNORED_DIRS
            and not (current_path / name).is_symlink()
        )

        level = len(current_path.relative_to(root).parts)

        entries = [
            root.name + "/" if level == 0
            else "    " * (level - 1) + current_path.name + "/"
        ]

        for filename in sorted(filenames, key=str.lower):
            if filename.lower().startswith(".env"):
                continue

            if filename.lower() in IGNORED_FILES:
                continue

            file_path = current_path / filename

            if file_path.suffix.lower() in IGNORED_SUFFIXES:
                continue

            if file_path.is_symlink() or not file_path.is_file():
                continue

            entries.append("    " * level + filename)

        for entry in entries:
            if total_chars + len(entry) + 1 > max_chars:
                truncated = True
                break

            lines.append(entry)
            total_chars += len(entry) + 1

        if truncated:
            break

    if truncated:
        lines.append("... (tree truncated)")

    return "\n".join(lines)
