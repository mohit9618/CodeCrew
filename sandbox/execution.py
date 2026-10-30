
import subprocess

from .sandbox_config import (
    SANDBOX_IMAGE,
    CPU_LIMIT,
    MEMORY_LIMIT,
    TIMEOUT_SECONDS
)


def run_tests(workspace: str, test_path: str = "tests"):
    command = [
        "docker", "run",
        "--rm",
        "--network=none",
        "--cpus", str(CPU_LIMIT),
        "--memory", MEMORY_LIMIT,
        "--memory-swap", MEMORY_LIMIT,
        "--pids-limit", "100",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        f"type=bind,source={workspace},target=/workspace,readonly",
        "--workdir", "/workspace",
        "--user", "runner",
        SANDBOX_IMAGE,
        "python", "-m", "pytest",
        "-p", "no:cacheprovider",
        "--import-mode=prepend",
        test_path,
        "-v"
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False
        }

    except subprocess.TimeoutExpired as error:
        return {
            "exit_code": -1,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "timed_out": True
        }

    except FileNotFoundError:
        raise RuntimeError("Docker executable was not found")
