"""Run a project's install and check steps inside the Docker sandbox.

Two kinds of container are used, both locked down (non-root, CPU/memory/pid
limits, read-only root filesystem):

* install - network ON. Installs the project's dependencies into the
  workspace (.codecrew/venv for Python, node_modules for Node). Skipped
  when the dependency files have not changed since the last install.
* check - network OFF. Runs tests, lint, type check and build.

The workspace is mounted at /workspace in both, so paths inside the
virtualenv stay valid between them.
"""
import json
import subprocess
import uuid
from pathlib import Path

from .sandbox_config import (
    CPU_LIMIT,
    MAX_OUTPUT_CHARS,
    MEMORY_LIMIT,
    PIDS_LIMIT,
    SANDBOX_IMAGE,
    TMPFS_SIZE,
)
from .stack import detect_stack, plan_checks, plan_install

INSTALL_MARKER = ".codecrew/install.json"

# Files and folders CodeCrew creates inside the cloned repository. They
# must never end up in a commit.
GIT_EXCLUDES = (".codecrew/",)


def _tail(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    text = text or ""

    if len(text) <= limit:
        return text

    return "...(output truncated)\n" + text[-limit:]


def _as_text(value) -> str:
    # TimeoutExpired carries bytes even when subprocess ran in text mode.
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return value


def ensure_git_excludes(workspace: str) -> None:
    """Keep CodeCrew's own files out of `git status` in the workspace."""
    exclude = Path(workspace) / ".git" / "info" / "exclude"

    if not exclude.parent.is_dir():
        return

    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    missing = [line for line in GIT_EXCLUDES if line not in existing]

    if missing:
        with exclude.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")

            handle.write("\n".join(missing) + "\n")


def build_docker_command(
    workspace: str,
    project_path: str,
    script: str,
    *,
    name: str,
    network: bool,
    writable: bool,
) -> list[str]:
    workdir = (
        "/workspace" if project_path in ("", ".")
        else f"/workspace/{project_path}"
    )

    mount = f"type=bind,source={workspace},target=/workspace"

    if not writable:
        mount += ",readonly"

    # The cloned repository's own code runs in these containers (install
    # scripts, tests, build). It must not be able to rewrite .git: a planted
    # hook or core.fsmonitor setting would later run on the host when the
    # publisher commits. So .git is always mounted read-only, even when
    # the rest of the workspace is writable.
    git_dir = Path(workspace) / ".git"

    command = [
        "docker", "run",
        "--rm",
        "--name", name,
        "--cpus", str(CPU_LIMIT),
        "--memory", MEMORY_LIMIT,
        "--memory-swap", MEMORY_LIMIT,
        "--pids-limit", str(PIDS_LIMIT),
        "--read-only",
        "--tmpfs", f"/tmp:rw,nosuid,size={TMPFS_SIZE}",
        "--mount", mount,
        *(
            ["--mount", f"type=bind,source={git_dir},target=/workspace/.git,readonly"]
            if git_dir.is_dir() else []
        ),
        "--workdir", workdir,
        "--user", "runner",
        "-e", "HOME=/tmp",
        "-e", "PIP_CACHE_DIR=/tmp/pip-cache",
        "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1",
        "-e", "npm_config_cache=/tmp/npm-cache",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
    ]

    if not network:
        command += ["--network=none"]

    command += [SANDBOX_IMAGE, "sh", "-c", script]

    return command


def _run_container(
    workspace: str,
    project_path: str,
    script: str,
    *,
    network: bool,
    writable: bool,
    timeout: int,
) -> dict:
    name = f"codecrew-{uuid.uuid4().hex[:12]}"

    command = build_docker_command(
        workspace, project_path, script,
        name=name, network=network, writable=writable,
    )

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        outcome = {
            "exit_code": result.returncode,
            "output": _tail(result.stdout + result.stderr),
            "timed_out": False,
        }

        _raise_if_docker_failed(outcome)

        return outcome

    except subprocess.TimeoutExpired as error:
        # Killing the docker client does not stop the container.
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            check=False,
        )

        return {
            "exit_code": -1,
            "output": _tail(
                _as_text(error.stdout) + _as_text(error.stderr)
            ),
            "timed_out": True,
        }

    except FileNotFoundError:
        raise RuntimeError("Docker executable was not found")


# Signs that `docker run` itself failed (daemon down, image missing), as
# opposed to the command inside the container failing.
DOCKER_FAILURE_MARKERS = (
    "Cannot connect to the Docker daemon",
    "error during connect",
    "Unable to find image",
    "pull access denied",
)


def _raise_if_docker_failed(result: dict) -> None:
    """Fail loudly when the sandbox could not start at all.

    Otherwise the error text would be recorded as a failed test run and the
    developers would be asked to "fix" a problem that is not in the code.
    """
    output = result["output"]

    # 125 is docker's own "the run command failed" exit code.
    if result["exit_code"] == 125 or any(
        marker in output for marker in DOCKER_FAILURE_MARKERS
    ):
        raise RuntimeError(
            "Docker could not start the sandbox. Check that Docker is "
            "running and that the image is built "
            f"(docker build -t {SANDBOX_IMAGE} sandbox/).\n"
            + output.strip()[-600:]
        )


def _install_is_current(workspace: str, project: dict, fingerprint: str) -> bool:
    marker = Path(workspace) / project["path"] / INSTALL_MARKER

    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    return data.get("fingerprint") == fingerprint


def _write_install_marker(workspace: str, project: dict, fingerprint: str) -> None:
    marker = Path(workspace) / project["path"] / INSTALL_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"fingerprint": fingerprint}),
        encoding="utf-8",
    )


def _entry(project, name, blocking, result, ok, **extra) -> dict:
    return {
        "project": project["path"],
        "name": name,
        "blocking": blocking,
        "ok": ok,
        "exit_code": result["exit_code"],
        "timed_out": result["timed_out"],
        "output": result["output"],
        **extra,
    }


def _install(workspace: str, project: dict) -> dict | None:
    plan = plan_install(workspace, project)

    if plan is None:
        return None

    if _install_is_current(workspace, project, plan["fingerprint"]):
        return _entry(
            project, "install", True,
            {"exit_code": 0, "output": "dependencies already installed",
             "timed_out": False},
            True,
        )

    script = "set -e\n" + "\n".join(plan["commands"])

    result = _run_container(
        workspace, project["path"], script,
        network=True, writable=True, timeout=plan["timeout"],
    )

    ok = result["exit_code"] == 0

    if ok:
        _write_install_marker(workspace, project, plan["fingerprint"])

    return _entry(project, "install", True, result, ok)


def format_report(entries: list[dict]) -> str:
    """Readable text of every step, for the tester agent and the developers."""
    sections = []

    for entry in entries:
        if entry["ok"]:
            status = "PASSED"
        elif entry["blocking"]:
            status = "FAILED"
        else:
            status = "WARNING"

        if entry.get("skipped"):
            status = "SKIPPED"

        header = f"=== [{entry['project']}] {entry['name']}: {status}"

        if entry["timed_out"]:
            header += " (timed out)"

        header += " ==="

        sections.append(f"{header}\n{entry['output'].strip()}".rstrip())

    return "\n\n".join(sections)


def run_checks(workspace: str, changed_files: list[str] | None = None) -> dict:
    """Install dependencies and run every check for each project.

    Returns the same keys the tester node has always used (exit_code,
    stdout, stderr, timed_out) plus `passed`, `checks` and `stack`.
    `passed` is True when every *blocking* step succeeded; lint and type
    check problems are reported but do not fail the run.
    """
    workspace = str(Path(workspace).resolve())
    ensure_git_excludes(workspace)

    stack = detect_stack(workspace)
    entries = []

    for project in stack["projects"]:
        install = _install(workspace, project)

        if install is not None:
            entries.append(install)

            if not install["ok"]:
                # Without dependencies nothing else can run meaningfully.
                continue

        for check in plan_checks(project, changed_files):
            result = _run_container(
                workspace, project["path"], check["command"],
                network=False,
                writable=check["writable"],
                timeout=check["timeout"],
            )

            ok = (
                not result["timed_out"]
                and result["exit_code"] in check["ok_exit_codes"]
            )

            entries.append(
                _entry(project, check["name"], check["blocking"], result, ok)
            )

    blocking_failures = [
        entry for entry in entries if entry["blocking"] and not entry["ok"]
    ]

    report = format_report(entries)

    if not entries:
        report = (
            "No runnable Python or Node project was detected, so no "
            "checks were run."
        )

    return {
        "passed": not blocking_failures,
        "exit_code": (
            blocking_failures[0]["exit_code"] if blocking_failures else 0
        ),
        "timed_out": any(entry["timed_out"] for entry in blocking_failures),
        "stdout": report,
        "stderr": "",
        "checks": entries,
        "no_checks": not entries,
        "stack": stack["summary"],
    }
