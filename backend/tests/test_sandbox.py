"""Stack detection, check planning and the Docker runner (Docker is faked)."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from sandbox import execution
from sandbox.stack import (
    detect_stack,
    is_test_file,
    plan_checks,
    plan_install,
)


def write(root, relative, text=""):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def python_project(tmp_path, requirements="fastapi\nuvicorn\n"):
    write(tmp_path, "requirements.txt", requirements)
    write(tmp_path, "app/main.py", "print('hi')\n")
    return tmp_path


# --------------------------------------------------------------- detection

def test_detects_python_project_and_frameworks(tmp_path):
    stack = detect_stack(str(python_project(tmp_path)))

    assert stack["languages"] == ["python"]
    project = stack["projects"][0]
    assert project["path"] == "."
    assert "fastapi" in project["frameworks"]
    assert project["requirement_files"] == ["requirements.txt"]
    assert "fastapi" in stack["summary"]


def test_detects_node_project_scripts_and_frameworks(tmp_path):
    write(tmp_path, "package.json", json.dumps({
        "dependencies": {"react": "^18"},
        "devDependencies": {"vite": "^5", "typescript": "^5"},
        "scripts": {"test": "vitest run", "build": "vite build"},
    }))
    write(tmp_path, "tsconfig.json", "{}")

    project = detect_stack(str(tmp_path))["projects"][0]

    assert project["kind"] == "node"
    assert {"react", "vite", "typescript"} <= set(project["frameworks"])
    assert project["has_typescript"] is True


def test_detects_backend_and_frontend_subprojects(tmp_path):
    write(tmp_path, "backend/requirements.txt", "flask\n")
    write(tmp_path, "frontend/package.json", json.dumps({
        "dependencies": {"vue": "^3"}
    }))

    stack = detect_stack(str(tmp_path))
    found = {(p["path"], p["kind"]) for p in stack["projects"]}

    assert found == {("backend", "python"), ("frontend", "node")}
    assert stack["languages"] == ["node", "python"]


def test_repository_without_project_files_has_no_projects(tmp_path):
    write(tmp_path, "README.md", "# docs only")

    stack = detect_stack(str(tmp_path))

    assert stack["projects"] == []
    assert "No runnable" in stack["summary"]


def test_broken_manifests_do_not_crash_detection(tmp_path):
    write(tmp_path, "package.json", "{not json")
    write(tmp_path, "pyproject.toml", "= broken")

    stack = detect_stack(str(tmp_path))

    assert {p["kind"] for p in stack["projects"]} == {"node", "python"}


# -------------------------------------------------------------- install plan

def test_python_install_plan(tmp_path):
    write(tmp_path, "requirements.txt", "flask\n")
    write(tmp_path, "pyproject.toml",
          "[project]\nname='x'\nversion='1'\n"
          "[dependency-groups]\ndev=['pytest']\n")
    project = detect_stack(str(tmp_path))["projects"][0]

    plan = plan_install(str(tmp_path), project)
    joined = "\n".join(plan["commands"])

    assert (
        "python -m venv --copies --system-site-packages .codecrew/venv"
        in joined
    )
    assert "pip install -r requirements.txt" in joined
    assert "pip install -e ." in joined
    assert "--group dev || true" in joined


def test_install_fingerprint_changes_with_dependencies(tmp_path):
    python_project(tmp_path)
    project = detect_stack(str(tmp_path))["projects"][0]
    first = plan_install(str(tmp_path), project)["fingerprint"]

    write(tmp_path, "requirements.txt", "fastapi\nrequests\n")
    second = plan_install(str(tmp_path), project)["fingerprint"]

    assert first != second


def test_node_install_uses_ci_only_with_lockfile(tmp_path):
    write(tmp_path, "package.json", "{}")
    project = detect_stack(str(tmp_path))["projects"][0]
    assert plan_install(str(tmp_path), project)["commands"] == [
        "npm install --no-audit --no-fund"
    ]

    write(tmp_path, "package-lock.json", "{}")
    project = detect_stack(str(tmp_path))["projects"][0]
    assert plan_install(str(tmp_path), project)["commands"][0].startswith(
        "npm ci"
    )


def test_python_project_with_nothing_to_install_still_gets_a_venv(tmp_path):
    # The checks run the venv's interpreter, so it must always exist.
    write(tmp_path, "Pipfile", "")
    project = detect_stack(str(tmp_path))["projects"][0]
    project["installable"] = False

    plan = plan_install(str(tmp_path), project)

    assert plan["commands"] == [
        "python -m venv --copies --system-site-packages .codecrew/venv"
    ]


def test_editable_install_failure_is_not_fatal(tmp_path):
    write(tmp_path, "pyproject.toml", "[project]\nname='x'\nversion='1'\n")
    project = detect_stack(str(tmp_path))["projects"][0]

    commands = plan_install(str(tmp_path), project)["commands"]
    editable = next(c for c in commands if "install -e ." in c)

    assert "||" in editable  # a flat-layout app can still run its tests


# -------------------------------------------------------------- check plans

def test_is_test_file():
    assert is_test_file("tests/test_api.py")
    assert is_test_file("pkg/api_test.py")
    assert is_test_file("src/App.test.tsx")
    assert not is_test_file("app/main.py")
    assert not is_test_file("src/App.tsx")


def test_python_checks_target_changed_tests_and_compile_changed_files(tmp_path):
    python_project(tmp_path)
    project = detect_stack(str(tmp_path))["projects"][0]

    checks = {
        c["name"]: c for c in plan_checks(
            project, ["app/main.py", "tests/test_main.py"]
        )
    }

    assert checks["tests"]["command"].endswith("tests/test_main.py")
    assert checks["tests"]["blocking"] is True
    assert 5 in checks["tests"]["ok_exit_codes"]  # nothing collected
    assert checks["compile"]["blocking"] is True
    assert "app/main.py" in checks["compile"]["command"]
    assert checks["lint"]["blocking"] is False


def test_python_checks_run_whole_suite_when_no_test_was_changed(tmp_path):
    python_project(tmp_path)
    project = detect_stack(str(tmp_path))["projects"][0]

    tests = next(
        c for c in plan_checks(project, ["app/main.py"])
        if c["name"] == "tests"
    )

    assert tests["command"].endswith("pytest -p no:cacheprovider -q")


def test_changed_files_only_count_for_their_own_project(tmp_path):
    write(tmp_path, "backend/requirements.txt", "flask\n")
    project = detect_stack(str(tmp_path))["projects"][0]

    names = [
        c["name"] for c in plan_checks(project, ["frontend/app.py"])
    ]

    assert "compile" not in names and "lint" not in names

    commands = " ".join(
        c["command"] for c in plan_checks(project, ["backend/api/x.py"])
    )
    assert "api/x.py" in commands and "backend/api" not in commands


def test_node_checks_skip_placeholder_test_script(tmp_path):
    write(tmp_path, "package.json", json.dumps({
        "scripts": {
            "test": 'echo "Error: no test specified" && exit 1',
            "lint": "eslint .",
            "build": "vite build",
        }
    }))
    project = detect_stack(str(tmp_path))["projects"][0]

    checks = {c["name"]: c for c in plan_checks(project)}

    assert "tests" not in checks
    assert checks["lint"]["blocking"] is False
    assert checks["build"]["blocking"] is True


def test_node_typecheck_never_downloads(tmp_path):
    write(tmp_path, "package.json", "{}")
    write(tmp_path, "tsconfig.json", "{}")
    project = detect_stack(str(tmp_path))["projects"][0]

    typecheck = next(c for c in plan_checks(project) if c["name"] == "typecheck")

    assert "--no-install" in typecheck["command"]


# ------------------------------------------------------------ docker runner

class FakeDocker:
    """Stands in for subprocess.run and records every docker call."""

    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        script = command[-1] if command[0] == "docker" and "run" in command else ""

        for needle, (code, out) in self.results.items():
            if needle in script:
                return SimpleNamespace(returncode=code, stdout=out, stderr="")

        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def run_calls(self):
        return [c for c in self.calls if c[:2] == ["docker", "run"]]


@pytest.fixture
def fake_docker(monkeypatch):
    def install(results=None):
        fake = FakeDocker(results)
        monkeypatch.setattr(execution.subprocess, "run", fake)
        return fake

    return install


def test_install_has_network_and_checks_do_not(tmp_path, fake_docker):
    python_project(tmp_path)
    fake = fake_docker()

    result = execution.run_checks(str(tmp_path), ["tests/test_main.py"])

    runs = fake.run_calls()
    install, checks = runs[0], runs[1:]

    assert "--network=none" not in install
    assert all("--network=none" in command for command in checks)
    assert result["passed"] is True
    assert result["exit_code"] == 0
    assert [c["name"] for c in result["checks"]][0] == "install"


def test_checks_mount_workspace_read_only_for_python(tmp_path, fake_docker):
    python_project(tmp_path)
    fake = fake_docker()

    execution.run_checks(str(tmp_path), ["tests/test_main.py"])

    tests_run = next(c for c in fake.run_calls() if "pytest" in c[-1])
    mount = tests_run[tests_run.index("--mount") + 1]

    assert mount.endswith(",readonly")


def test_install_is_skipped_when_dependencies_unchanged(tmp_path, fake_docker):
    python_project(tmp_path)
    fake = fake_docker()

    execution.run_checks(str(tmp_path))
    first = len(fake.run_calls())
    execution.run_checks(str(tmp_path))
    second = len(fake.run_calls()) - first

    assert second == first - 1  # same checks, no second install

    write(tmp_path, "requirements.txt", "fastapi\nrequests\n")
    execution.run_checks(str(tmp_path))
    third = len(fake.run_calls()) - first - second

    assert third == first  # dependencies changed: installed again


def test_failing_test_fails_the_run_and_reports_output(tmp_path, fake_docker):
    python_project(tmp_path)
    fake_docker({"pytest": (1, "FAILED tests/test_main.py::test_x")})

    result = execution.run_checks(str(tmp_path))

    assert result["passed"] is False
    assert result["exit_code"] == 1
    assert "FAILED tests/test_main.py::test_x" in result["stdout"]
    assert "tests: FAILED" in result["stdout"]


def test_no_tests_collected_is_not_a_failure(tmp_path, fake_docker):
    python_project(tmp_path)
    fake_docker({"pytest": (5, "no tests ran")})

    assert execution.run_checks(str(tmp_path))["passed"] is True


def test_lint_problems_are_warnings_not_failures(tmp_path, fake_docker):
    python_project(tmp_path)
    fake_docker({"ruff": (1, "F401 unused import")})

    result = execution.run_checks(str(tmp_path), ["app/main.py"])

    assert result["passed"] is True
    assert "lint: WARNING" in result["stdout"]


def test_failed_install_stops_that_project(tmp_path, fake_docker):
    python_project(tmp_path)
    fake = fake_docker({"pip install": (1, "no matching distribution")})

    result = execution.run_checks(str(tmp_path))

    assert result["passed"] is False
    assert len(fake.run_calls()) == 1  # nothing ran after the install
    assert "no matching distribution" in result["stdout"]


def test_repository_without_a_project_reports_no_checks(tmp_path, fake_docker):
    write(tmp_path, "README.md", "docs")
    fake = fake_docker()

    result = execution.run_checks(str(tmp_path))

    assert result["passed"] is True
    assert result["no_checks"] is True
    assert fake.run_calls() == []
    assert "No runnable" in result["stdout"]


def test_timeout_removes_the_container(tmp_path, monkeypatch):
    python_project(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

        if command[:2] == ["docker", "run"] and "pytest" in command[-1]:
            raise subprocess.TimeoutExpired(command, 1, output=b"partial")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    result = execution.run_checks(str(tmp_path))

    assert result["passed"] is False
    assert result["timed_out"] is True
    assert result["exit_code"] == -1
    assert any(c[:3] == ["docker", "rm", "-f"] for c in calls)
    assert "partial" in result["stdout"]


def test_missing_docker_raises_clear_error(tmp_path, monkeypatch):
    python_project(tmp_path)

    def fake_run(command, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Docker executable"):
        execution.run_checks(str(tmp_path))


def test_codecrew_folder_is_excluded_from_git(tmp_path):
    (tmp_path / ".git" / "info").mkdir(parents=True)

    execution.ensure_git_excludes(str(tmp_path))
    execution.ensure_git_excludes(str(tmp_path))  # idempotent

    text = (tmp_path / ".git" / "info" / "exclude").read_text()

    assert text.count(".codecrew/") == 1


def test_long_output_keeps_the_tail():
    text = "a" * 100 + "SUMMARY"

    assert execution._tail(text, 20).endswith("SUMMARY")
    assert execution._tail("short", 20) == "short"
