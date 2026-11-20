"""Detect a repository's tech stack and plan how to install and check it.

Everything here is plain Python over the filesystem (no Docker, no LLM), so
it is fast and unit-testable. `sandbox.execution` runs the plans.

Results are JSON-serialisable dicts because they are stored in the LangGraph
state, which is checkpointed to PostgreSQL.
"""
import hashlib
import json
import shlex
import tomllib
from pathlib import Path, PurePosixPath

# Directories (besides the repository root) that commonly hold a
# self-contained backend or frontend project.
SUBPROJECT_DIRS = (
    "backend", "server", "api", "frontend", "client", "web", "app",
)

PYTHON_MARKERS = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile",
)

# Frameworks we recognise, mapped to the dependency name that reveals them.
PYTHON_FRAMEWORKS = {
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "sqlalchemy": "sqlalchemy",
    "alembic": "alembic",
    "pydantic": "pydantic",
}
NODE_FRAMEWORKS = {
    "react": "react",
    "next": "next",
    "vue": "vue",
    "svelte": "svelte",
    "@angular/core": "angular",
    "express": "express",
    "vite": "vite",
    "typescript": "typescript",
    "prisma": "prisma",
    "tailwindcss": "tailwindcss",
}

# `npm init` writes this placeholder test script; it always fails.
NPM_PLACEHOLDER_TEST = "no test specified"

TEST_FILE_SUFFIXES = (
    ".test.js", ".test.jsx", ".test.ts", ".test.tsx",
    ".spec.js", ".spec.jsx", ".spec.ts", ".spec.tsx",
)

# Per-step time limits in seconds.
INSTALL_TIMEOUT = 600
TEST_TIMEOUT = 180
LINT_TIMEOUT = 120
BUILD_TIMEOUT = 300


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(_read(path))
    except ValueError:
        return {}

    return data if isinstance(data, dict) else {}


def _load_toml(path: Path) -> dict:
    try:
        return tomllib.loads(_read(path))
    except tomllib.TOMLDecodeError:
        return {}


def _requirement_files(directory: Path) -> list[str]:
    found = sorted(p.name for p in directory.glob("requirements*.txt"))

    requirements_dir = directory / "requirements"

    if requirements_dir.is_dir():
        found += sorted(
            f"requirements/{p.name}" for p in requirements_dir.glob("*.txt")
        )

    return found


def _python_dependency_text(directory: Path, requirement_files: list[str]) -> str:
    parts = [_read(directory / name) for name in requirement_files]
    parts.append(_read(directory / "pyproject.toml"))
    parts.append(_read(directory / "setup.cfg"))
    parts.append(_read(directory / "setup.py"))

    return "\n".join(parts).lower()


def _detect_python(directory: Path, rel: str) -> dict | None:
    if not any((directory / marker).exists() for marker in PYTHON_MARKERS):
        return None

    requirement_files = _requirement_files(directory)
    dependency_text = _python_dependency_text(directory, requirement_files)

    pyproject = _load_toml(directory / "pyproject.toml")

    # Installable when it declares how to build itself: a [project] table
    # (PEP 621), a build backend (this includes Poetry), or setup.py/cfg.
    installable = (
        "project" in pyproject
        or "build-system" in pyproject
        or (directory / "setup.py").exists()
        or (directory / "setup.cfg").exists()
    )

    return {
        "path": rel,
        "kind": "python",
        "frameworks": sorted(
            label for name, label in PYTHON_FRAMEWORKS.items()
            if name in dependency_text
        ),
        "requirement_files": requirement_files,
        "installable": bool(installable),
        "dependency_groups": sorted(
            pyproject.get("dependency-groups", {}) or {}
        ),
        "scripts": {},
        "has_typescript": False,
    }


def _detect_node(directory: Path, rel: str) -> dict | None:
    package_json = directory / "package.json"

    if not package_json.exists():
        return None

    package = _load_json(package_json)

    dependencies = {
        **(package.get("dependencies") or {}),
        **(package.get("devDependencies") or {}),
    }

    scripts = package.get("scripts") or {}

    if not isinstance(scripts, dict):
        scripts = {}

    return {
        "path": rel,
        "kind": "node",
        "frameworks": sorted(
            label for name, label in NODE_FRAMEWORKS.items()
            if name in dependencies
        ),
        "requirement_files": [],
        "installable": True,
        "dependency_groups": [],
        "scripts": {
            name: value for name, value in scripts.items()
            if isinstance(value, str)
        },
        "has_typescript": (directory / "tsconfig.json").exists(),
        "has_lockfile": (directory / "package-lock.json").exists(),
    }


def detect_stack(workspace: str) -> dict:
    """Find the runnable projects in a repository and summarise the stack."""
    root = Path(workspace).resolve()

    candidates = [(root, ".")]
    candidates += [
        (root / name, name) for name in SUBPROJECT_DIRS
        if (root / name).is_dir() and not (root / name).is_symlink()
    ]

    projects = []

    for directory, rel in candidates:
        for detector in (_detect_python, _detect_node):
            project = detector(directory, rel)

            if project:
                projects.append(project)

    languages = sorted({project["kind"] for project in projects})

    return {
        "projects": projects,
        "languages": languages,
        "summary": describe_stack(projects),
    }


def describe_stack(projects: list[dict]) -> str:
    """One short line per project, for prompts."""
    if not projects:
        return "No runnable Python or Node project detected."

    lines = []

    for project in projects:
        frameworks = ", ".join(project["frameworks"]) or "no known framework"
        extras = []

        if project["kind"] == "node":
            names = [
                name for name in ("test", "lint", "build")
                if name in project["scripts"]
            ]

            if names:
                extras.append("npm scripts: " + ", ".join(names))

            if project["has_typescript"]:
                extras.append("TypeScript")

        suffix = f" ({'; '.join(extras)})" if extras else ""
        lines.append(
            f"- {project['path']}: {project['kind']} - {frameworks}{suffix}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Installing dependencies
# ---------------------------------------------------------------------------

def _fingerprint(directory: Path, project: dict) -> str:
    """Hash of the files that decide which dependencies are installed."""
    names = [
        "pyproject.toml", "setup.py", "setup.cfg", "Pipfile",
        "package.json", "package-lock.json",
        *project["requirement_files"],
    ]

    digest = hashlib.sha256()

    for name in names:
        path = directory / name

        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())

    return digest.hexdigest()


# Interpreter inside the per-project virtualenv created by plan_install.
VENV_PYTHON = ".codecrew/venv/bin/python"

# Syntax-checks files without writing .pyc files (the workspace is mounted
# read-only while checks run).
COMPILE_SCRIPT = (
    "import sys\n"
    "failed = False\n"
    "for name in sys.argv[1:]:\n"
    "    try:\n"
    "        compile(open(name, encoding='utf-8').read(), name, 'exec')\n"
    "    except SyntaxError as error:\n"
    "        print(f'{name}:{error.lineno}: {error.msg}')\n"
    "        failed = True\n"
    "sys.exit(1 if failed else 0)\n"
)


def plan_install(workspace: str, project: dict) -> dict | None:
    """Commands that install a project's dependencies, or None if nothing.

    The install step needs network access; the check steps do not. Python
    dependencies go into a virtualenv inside the project (.codecrew/venv)
    that can see the image's pytest/ruff. The project itself is installed
    in editable mode, so later edits are picked up without reinstalling.
    """
    directory = (Path(workspace).resolve() / project["path"]).resolve()
    commands = []

    if project["kind"] == "python":
        pip = ".codecrew/venv/bin/pip"

        # Always create the virtualenv: the checks run its interpreter, even
        # for a project that has nothing to install. --copies because
        # symlinks are unreliable on bind mounts from a Windows drive.
        commands.append(
            "python -m venv --copies --system-site-packages .codecrew/venv"
        )

        for name in project["requirement_files"]:
            commands.append(f"{pip} install -r {shlex.quote(name)}")

        if project["installable"]:
            # Not fatal: apps with a flat layout and several top-level
            # packages cannot be installed as a package, but their tests
            # may still run from the source tree.
            commands.append(
                f"{pip} install -e . || echo 'WARNING: editable install "
                "failed; continuing without it'"
            )

        # PEP 735 dependency groups (e.g. a "dev" group). Best effort: older
        # pip versions do not know --group.
        for group in project["dependency_groups"]:
            commands.append(
                f"{pip} install --group {shlex.quote(group)} || true"
            )

    elif project["kind"] == "node":
        if project["has_lockfile"]:
            commands.append(
                "npm ci --no-audit --no-fund "
                "|| npm install --no-audit --no-fund"
            )
        else:
            commands.append("npm install --no-audit --no-fund")

    else:
        return None

    return {
        "project": project["path"],
        "kind": project["kind"],
        "commands": commands,
        "fingerprint": _fingerprint(directory, project),
        "timeout": INSTALL_TIMEOUT,
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def is_test_file(path: str) -> bool:
    name = PurePosixPath(path).name

    if name.endswith(".py"):
        return name.startswith("test_") or name.endswith("_test.py")

    return name.endswith(TEST_FILE_SUFFIXES)


def _files_in_project(project: dict, changed_files: list[str], suffixes) -> list[str]:
    """Changed files inside a project, relative to the project directory."""
    prefix = "" if project["path"] == "." else project["path"].rstrip("/") + "/"
    result = []

    for path in changed_files:
        path = path.replace("\\", "/")

        if not path.startswith(prefix):
            continue

        relative = path[len(prefix):]

        if relative.endswith(suffixes):
            result.append(relative)

    return result


def _check(name, command, *, blocking, writable, timeout, ok_exit_codes=(0,)):
    return {
        "name": name,
        "command": command,
        "blocking": blocking,
        "writable": writable,
        "timeout": timeout,
        "ok_exit_codes": list(ok_exit_codes),
    }


def _plan_python_checks(project: dict, changed: list[str]) -> list[dict]:
    changed_py = _files_in_project(project, changed, (".py",))
    changed_tests = [path for path in changed_py if is_test_file(path)]

    checks = []

    # "Build": syntax-check the changed files to catch errors fast.
    if changed_py:
        files = " ".join(shlex.quote(path) for path in changed_py)
        checks.append(_check(
            "compile",
            f"{VENV_PYTHON} -c {shlex.quote(COMPILE_SCRIPT)} {files}",
            blocking=True, writable=False, timeout=LINT_TIMEOUT,
        ))

    # Run the tests the developers wrote; if there are none, the whole
    # suite. Exit code 5 means "no tests collected", which is not a failure.
    targets = " ".join(shlex.quote(path) for path in changed_tests)
    checks.append(_check(
        "tests",
        f"{VENV_PYTHON} -m pytest -p no:cacheprovider -q {targets}".strip(),
        blocking=True, writable=False, timeout=TEST_TIMEOUT,
        ok_exit_codes=(0, 5),
    ))

    if changed_py:
        files = " ".join(shlex.quote(path) for path in changed_py)
        checks.append(_check(
            "lint",
            f"ruff check --no-cache {files}",
            blocking=False, writable=False, timeout=LINT_TIMEOUT,
        ))

    return checks


def _plan_node_checks(project: dict) -> list[dict]:
    scripts = project["scripts"]
    checks = []

    test_script = scripts.get("test", "")

    if test_script and NPM_PLACEHOLDER_TEST not in test_script:
        checks.append(_check(
            "tests",
            "CI=true npm test --silent",
            blocking=True, writable=True, timeout=TEST_TIMEOUT,
        ))

    if "lint" in scripts:
        checks.append(_check(
            "lint", "npm run lint --silent",
            blocking=False, writable=True, timeout=LINT_TIMEOUT,
        ))

    if project["has_typescript"]:
        # --no-install: the sandbox has no network, so never try to fetch.
        checks.append(_check(
            "typecheck", "npx --no-install tsc --noEmit",
            blocking=False, writable=True, timeout=LINT_TIMEOUT,
        ))

    if "build" in scripts:
        checks.append(_check(
            "build", "npm run build --silent",
            blocking=True, writable=True, timeout=BUILD_TIMEOUT,
        ))

    return checks


def plan_checks(project: dict, changed_files: list[str] | None = None) -> list[dict]:
    """Checks to run for one project. Blocking checks decide pass/fail."""
    changed = list(changed_files or [])

    if project["kind"] == "python":
        return _plan_python_checks(project, changed)

    if project["kind"] == "node":
        return _plan_node_checks(project)

    return []
