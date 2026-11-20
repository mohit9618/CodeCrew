"""What CodeCrew never shows to a model or lets an agent touch."""

IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".next", "dist", "build", "coverage",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea",
    # CodeCrew's own sandbox files (virtualenv, install marker).
    ".codecrew",
}

IGNORED_FILES = {
    ".npmrc", ".pypirc", ".netrc", "id_rsa",
    "id_ed25519", "credentials", "secrets.json",
    "service-account.json", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock"
}

IGNORED_SUFFIXES = {
    ".pem", ".key", ".p12", ".pfx", ".crt", ".cer",
    ".der", ".pyc", ".db", ".sqlite", ".zip", ".exe",
    ".dll", ".png", ".jpg", ".jpeg", ".gif", ".pdf",
    ".mp4", ".mp3"
}


def is_sensitive_file(name: str) -> bool:
    """Secrets and key material: never read or written by an agent."""
    lowered = name.lower()

    return (
        lowered.startswith(".env")
        or lowered in IGNORED_FILES - {
            "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
            "poetry.lock", "uv.lock",
        }
        or any(lowered.endswith(suffix) for suffix in (
            ".pem", ".key", ".p12", ".pfx"
        ))
    )
