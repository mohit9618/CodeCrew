"""The developer's change -> check -> fix loop, with a scripted fake model."""
import pytest
from pydantic import ValidationError

from backend.app.agents import roles
from backend.app.agents.developer import (
    MAX_ITERATIONS,
    DeveloperAgent,
    DeveloperOutput,
    FileChange,
    find_syntax_errors,
)


# -------------------------------------------------------------------- fakes

class ScriptedLLM:
    """Returns the prepared outputs one by one and records every prompt."""

    def __init__(self, outputs, budget=20_000):
        self.outputs = list(outputs)
        self.budget = budget
        self.prompts = []

    def prompt_char_budget(self, overhead_tokens=0):
        return max(0, self.budget - overhead_tokens * 4)

    def with_structured_output(self, schema, method=None):
        return self

    async def ainvoke(self, messages):
        self.prompts.append(messages)

        if not self.outputs:
            raise AssertionError("the model was called more often than scripted")

        return self.outputs.pop(0)


class FakeFilesystem:
    def __init__(self, workspace):
        self.workspace = str(workspace)

    async def create_directory(self, path):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)

    async def write_file(self, path, content):
        from pathlib import Path
        Path(path).write_bytes(content.encode("utf-8"))


def new_file(path, content, summary="x"):
    return FileChange(path=path, content=content, summary=summary)


def edit_file(path, search, replace):
    return FileChange(
        path=path, edits=[{"search": search, "replace": replace}], summary="x"
    )


def output(*changes, summary="done", **kwargs):
    return DeveloperOutput(summary=summary, changes=list(changes), **kwargs)


TASK = {
    "task_id": "T1",
    "title": "Add health endpoint",
    "description": "Create a GET /health route returning status healthy",
    "files_to_modify": ["app/main.py"],
    "files_to_read": [],
    "acceptance_criteria": ["GET /health returns 200"],
}


async def run(tmp_path, outputs, **kwargs):
    llm = ScriptedLLM(outputs)
    agent = DeveloperAgent("backend", llm=llm)

    result = await agent.run_task(
        kwargs.pop("task", TASK),
        task_text=kwargs.pop("task_text", "Implement the task"),
        workspace=str(tmp_path),
        filesystem=FakeFilesystem(tmp_path),
        **kwargs,
    )

    return result, llm


def seed(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text(
        "from flask import Flask\n\napp = Flask(__name__)\n", encoding="utf-8"
    )


# ------------------------------------------------------------------- schema

def test_change_needs_content_or_edits_but_not_both():
    with pytest.raises(ValidationError):
        FileChange(path="a.py")

    with pytest.raises(ValidationError):
        FileChange(path="a.py", content="x", edits=[{"search": "a", "replace": "b"}])

    assert FileChange(path="a.py", content="").content == ""  # empty file is valid


def test_output_defaults_make_minimal_json_valid():
    parsed = DeveloperOutput.model_validate({"changes": []})

    assert parsed.read_files == [] and parsed.run_tests is False


# --------------------------------------------------------------------- loop

async def test_creates_a_new_file_in_one_round(tmp_path):
    result, llm = await run(tmp_path, [
        output(new_file("app/health.py", "def health():\n    return 'ok'\n"))
    ])

    assert result["applied_files"] == ["app/health.py"]
    assert result["iterations"] == 1
    assert (tmp_path / "app" / "health.py").read_text().startswith("def health")
    assert len(llm.prompts) == 1


async def test_edits_an_existing_file_with_search_replace(tmp_path):
    seed(tmp_path)

    result, _ = await run(tmp_path, [
        output(edit_file(
            "app/main.py",
            "app = Flask(__name__)\n",
            "app = Flask(__name__)\n\n\n@app.get('/health')\ndef health():\n"
            "    return {'status': 'healthy'}\n",
        ))
    ])

    text = (tmp_path / "app" / "main.py").read_text()
    assert "from flask import Flask" in text       # rest of the file kept
    assert "@app.get('/health')" in text
    assert result["applied_files"] == ["app/main.py"]


async def test_failed_edit_is_explained_to_the_model_and_retried(tmp_path):
    seed(tmp_path)

    result, llm = await run(tmp_path, [
        output(edit_file("app/main.py", "app = Flask(__nam__)", "app = 1")),
        output(edit_file("app/main.py", "app = Flask(__name__)",
                         "app = Flask(__name__, static_folder=None)")),
    ])

    second_prompt = llm.prompts[1][1][1]
    assert "OBSERVATIONS FROM YOUR PREVIOUS STEP" in second_prompt
    assert "search text was not found" in second_prompt
    assert "static_folder=None" in (tmp_path / "app" / "main.py").read_text()
    assert result["iterations"] == 2


async def test_syntax_error_is_reported_back_and_fixed(tmp_path):
    result, llm = await run(tmp_path, [
        output(new_file("app/health.py", "def health(:\n    pass\n")),
        output(new_file("app/health.py", "def health():\n    pass\n")),
    ])

    assert "Syntax error: app/health.py:1" in llm.prompts[1][1][1]
    assert result["iterations"] == 2
    assert result["warnings"] == []


async def test_model_can_ask_to_read_more_files(tmp_path):
    seed(tmp_path)
    (tmp_path / "app" / "config.py").write_text("DEBUG = True\n")

    result, llm = await run(
        tmp_path,
        [
            output(read_files=["app/config.py"]),
            output(new_file("app/health.py", "x = 1\n")),
        ],
        task={**TASK, "files_to_modify": [], "description": "zzz", "title": "zzz"},
    )

    assert "DEBUG = True" not in llm.prompts[0][1][1]
    assert "DEBUG = True" in llm.prompts[1][1][1]
    assert "now shown: app/config.py" in llm.prompts[1][1][1]
    assert result["applied_files"] == ["app/health.py"]


async def test_failing_checks_go_back_to_the_model(tmp_path):
    calls = []

    def check_runner(workspace, files):
        calls.append(list(files))

        if len(calls) == 1:
            return {"passed": False, "stdout": "FAILED test_health - assert 500 == 200"}

        return {"passed": True, "stdout": "ok"}

    result, llm = await run(
        tmp_path,
        [
            output(new_file("app/health.py", "x = 1\n"), run_tests=True),
            output(new_file("app/health.py", "x = 2\n"), run_tests=True),
        ],
        check_runner=check_runner,
        applied_before=["earlier.py"],
    )

    assert "assert 500 == 200" in llm.prompts[1][1][1]
    assert len(calls) == 2
    assert "earlier.py" in calls[0]   # earlier tasks' files are checked too
    assert result["iterations"] == 2


async def test_checks_run_only_when_the_model_asks(tmp_path):
    def check_runner(workspace, files):
        raise AssertionError("checks must not run unless requested")

    await run(
        tmp_path, [output(new_file("a.py", "x = 1\n"))],
        check_runner=check_runner,
    )


async def test_no_changes_on_a_first_attempt_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="generated no changes"):
        await run(tmp_path, [output(summary="nothing to do")])


async def test_no_changes_is_fine_when_reworking_after_a_failure(tmp_path):
    result, _ = await run(
        tmp_path, [output(summary="already correct")], allow_no_changes=True
    )

    assert result["applied_files"] == []
    assert result["summary"] == "already correct"


async def test_blockers_stop_the_task(tmp_path):
    with pytest.raises(RuntimeError, match="blockers"):
        await run(tmp_path, [output(questions=["Which database?"])])


async def test_unsafe_paths_are_never_written(tmp_path):
    outside = tmp_path.parent / "escaped.py"

    result, llm = await run(tmp_path, [
        output(new_file("../escaped.py", "x = 1\n"),
               new_file(".env", "SECRET=1\n"),
               new_file("ok.py", "x = 1\n")),
        output(summary="nothing more"),
    ])

    assert not outside.exists()
    assert not (tmp_path / ".env").exists()
    assert result["applied_files"] == ["ok.py"]

    # The model is told why two of its changes were refused.
    feedback = llm.prompts[1][1][1]
    assert "Unsafe repository path" in feedback
    assert "sensitive file" in feedback


async def test_gives_up_after_max_iterations_with_warnings(tmp_path):
    result, _ = await run(tmp_path, [
        output(new_file("bad.py", "def (:\n")) for _ in range(MAX_ITERATIONS)
    ])

    assert result["iterations"] == MAX_ITERATIONS
    assert any("Syntax error" in warning for warning in result["warnings"])
    assert result["applied_files"] == ["bad.py"]


async def test_raises_when_nothing_could_be_applied(tmp_path):
    seed(tmp_path)

    with pytest.raises(RuntimeError, match="no usable changes"):
        await run(tmp_path, [
            output(edit_file("app/main.py", "does not exist", "x"))
            for _ in range(MAX_ITERATIONS)
        ])


async def test_editing_a_missing_file_tells_the_model_to_create_it(tmp_path):
    _, llm = await run(tmp_path, [
        output(edit_file("nope.py", "a", "b")),
        output(new_file("nope.py", "a = 1\n")),
    ])

    assert "does not exist" in llm.prompts[1][1][1]


async def test_relevant_files_are_found_without_being_named(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "health_service.py").write_text("def check_health(): ...\n")
    (tmp_path / "docs.md").write_text("unrelated")

    _, llm = await run(
        tmp_path,
        [output(new_file("app/x.py", "x = 1\n"))],
        task={**TASK, "files_to_modify": [], "files_to_read": []},
    )

    prompt = llm.prompts[0][1][1]
    assert "FILE: app/health_service.py" in prompt
    assert "unrelated" not in prompt


# ------------------------------------------------------------ prompt budget

async def test_prompt_fits_the_models_token_budget(tmp_path):
    llm = ScriptedLLM([output(new_file("a.py", "x"))], budget=24_000)
    agent = DeveloperAgent("backend", llm=llm)

    await agent.generate_changes(
        task="t" * 80_000, repo_tree="r" * 80_000, file_contents="f" * 80_000,
        observations="o" * 80_000,
    )

    system, human = llm.prompts[0][0][1], llm.prompts[0][1][1]

    assert len(system) + len(human) < 24_000


async def test_small_budget_models_get_smaller_prompts(tmp_path):
    big, small = ScriptedLLM([output()], budget=40_000), ScriptedLLM([output()], budget=10_000)

    for llm in (big, small):
        await DeveloperAgent("backend", llm=llm).generate_changes(
            "t" * 50_000, "r" * 50_000, "f" * 50_000
        )

    assert len(small.prompts[0][1][1]) < len(big.prompts[0][1][1])


async def test_errors_keep_their_tail_when_trimmed():
    llm = ScriptedLLM([output()], budget=6_000)

    await DeveloperAgent("backend", llm=llm).generate_changes(
        "task", "tree", "files",
        observations="noise " * 5_000 + "FAILED test_x - the important line",
    )

    assert "the important line" in llm.prompts[0][1][1]


# ---------------------------------------------------------- roles and prompt

def test_each_role_has_its_own_guidance():
    prompts = {
        role: roles.build_system_prompt(role)
        for role in ("backend", "database", "frontend")
    }

    assert "server-side" in prompts["backend"]
    assert "migration" in prompts["database"]
    assert "accessible" in prompts["frontend"]
    assert len(set(prompts.values())) == 3


def test_contract_and_stack_are_in_the_prompt_only_when_given():
    plain = roles.build_system_prompt("backend")
    full = roles.build_system_prompt(
        "frontend",
        stack_summary="- frontend: node - react",
        api_contract=["GET /items -> 200 [{id: int}]"],
        data_model=["items(id INTEGER PK)"],
    )

    assert "API CONTRACT (every developer" not in plain
    assert "PROJECT STACK" not in plain
    assert "- frontend: node - react" in full
    assert "GET /items -> 200" in full
    assert "items(id INTEGER PK)" in full


def test_unknown_role_falls_back_to_general():
    assert "Developer of CodeCrew" in roles.build_system_prompt("mystery")


def test_find_syntax_errors_only_looks_at_python(tmp_path):
    (tmp_path / "a.py").write_text("def (:\n")
    (tmp_path / "b.js").write_text("function (:\n")

    errors = find_syntax_errors(str(tmp_path), ["a.py", "b.js", "missing.py"])

    assert len(errors) == 1 and errors[0].startswith("a.py:1")
