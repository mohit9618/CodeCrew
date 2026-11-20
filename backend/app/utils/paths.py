"""Safe handling of repository-relative paths that come from an LLM."""
import re
from pathlib import Path, PurePosixPath

from backend.app.utils.ignore import IGNORED_DIRS, is_sensitive_file

# Agents may not create or change anything under these directories.
PROTECTED_DIRS = {".git", ".codecrew", "node_modules"}


def _is_protected(part: str) -> bool:
    """True if a path component names a protected directory.

    Case-insensitive and blind to trailing dots/spaces, because Windows
    treats `.GIT`, `.git.` and `.git ` as `.git`; also rejects the 8.3
    short name (`GIT~1`).
    """
    name = part.lower().rstrip(" .")

    return name in PROTECTED_DIRS or re.fullmatch(r"git~\d+", name) is not None


def safe_relative_path(path: str) -> PurePosixPath:
    """Validate a repository-relative path and return it normalised.

    Rejects absolute paths, drive letters, parent-directory escapes, empty
    paths, protected folders (.git, ...) and secret files (.env, keys).
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Empty file path.")

    normalised = path.strip().replace("\\", "/")
    rel_path = PurePosixPath(normalised)

    if (
        rel_path.is_absolute()
        or ".." in rel_path.parts
        or not rel_path.parts
        or ":" in normalised
    ):
        raise ValueError(f"Unsafe repository path: {path}")

    if any(_is_protected(part) for part in rel_path.parts):
        raise ValueError(f"Protected repository path: {path}")

    if is_sensitive_file(rel_path.name):
        raise ValueError(f"Refusing to touch a sensitive file: {path}")

    return rel_path


def resolve_in_workspace(workspace: str, path: str) -> Path:
    """Absolute path of `path` inside `workspace`, guaranteed to stay in it."""
    root = Path(workspace).resolve()
    rel_path = safe_relative_path(path)
    full_path = (root / Path(*rel_path.parts)).resolve()

    try:
        resolved = full_path.relative_to(root)
    except ValueError:
        raise ValueError(f"File path escapes workspace: {path}")

    # A symlink inside the repository (docs -> .git) passes the textual
    # check above; the resolved path reveals where it really points.
    if any(_is_protected(part) for part in resolved.parts):
        raise ValueError(f"Protected repository path: {path}")

    return full_path


def is_ignored_dir(name: str) -> bool:
    return name in IGNORED_DIRS
