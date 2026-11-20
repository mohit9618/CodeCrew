"""The development step end to end, with scripted agents and a fake disk."""
from pathlib import Path

from backend.app.agents.developer import DeveloperAgent, DeveloperOutput, FileChange
from backend.app.graph import developer_subgraph as subgraph


class ScriptedLLM:
    def __init__(self, role, log, files):
        self.role, self.log, self.files = role, log, files
        self.prompts = []

    def prompt_char_budget(self, overhead_tokens=0):
        return 40_000

    def with_structured_output(self, schema, method=None):
        return self

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        self.log.append(self.role)
        path, content = self.files[self.role]

        return DeveloperOutput(
            summary=f"{self.role} done",
            changes=[FileChange(path=path, content=content)],
        )


class FakeFilesystem:
    def __init__(self, workspace):
        self.workspace = str(workspace)
        self.connected = False

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def create_directory(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    async def write_file(self, path, content):
        Path(path).write_text(content, encoding="utf-8")


def task(task_id, role, files):
    return {
        "task_id": task_id, "title": f"task {task_id}",
        "description": "do it", "assigned_developer": role,
        "files_to_modify": files, "files_to_read": [],
        "dependencies": [], "acceptance_criteria": [],
    }


def test_tasks_follow_the_architects_batches():
    tasks = [task("T3", "frontend_developer", []),
             task("T1", "database_developer", []),
             task("T2", "backend_developer", []),
             task("T9", "backend_developer", [])]

    ordered = subgraph.order_tasks(tasks, [["T1"], ["T2", "T3"]])

    assert [t["task_id"] for t in ordered] == ["T1", "T2", "T3", "T9"]


def test_tasks_without_batches_keep_their_order():
    tasks = [task("B", "x", []), task("A", "x", [])]

    assert [t["task_id"] for t in subgraph.order_tasks(tasks, [])] == ["B", "A"]
    assert subgraph.order_tasks([], [["T1"]]) == []
    # An unknown id in the batches is skipped, not an error.
    assert subgraph.order_tasks(tasks, [["ZZ"], ["A"]])[0]["task_id"] == "A"


async def test_each_task_goes_to_its_role_in_dependency_order(
    tmp_path, monkeypatch
):
    log = []
    files = {
        "database_developer": ("db/schema.sql", "CREATE TABLE items (id INT);\n"),
        "backend_developer": ("app/api.py", "def items():\n    return []\n"),
        "frontend_developer": ("web/list.js", "fetch('/items');\n"),
    }
    llms = {}

    def make(role):
        def factory():
            llms[role] = ScriptedLLM(role, log, files)
            return DeveloperAgent(role.split("_")[0], llm=llms[role])
        return factory

    monkeypatch.setattr(subgraph, "DEVELOPER_ROLES", {
        role: make(role) for role in files
    })
    monkeypatch.setattr(subgraph, "FilesystemClient", FakeFilesystem)
    monkeypatch.setattr(
        subgraph, "run_checks",
        lambda *a: (_ for _ in ()).throw(AssertionError("not requested")),
    )

    state = {
        "workspace": str(tmp_path),
        # Deliberately listed out of order.
        "tasks": [
            task("T3", "frontend_developer", ["web/list.js"]),
            task("T1", "database_developer", ["db/schema.sql"]),
            task("T2", "backend_developer", ["app/api.py"]),
        ],
        "execution_batches": [["T1"], ["T2"], ["T3"]],
        "api_contract": ["GET /items -> 200 [{id: int}]"],
        "data_model": ["items(id INTEGER PK)"],
        "stack": {"summary": "- .: node - react"},
    }

    result = await subgraph.develop_node(state)

    assert log == ["database_developer", "backend_developer", "frontend_developer"]
    assert result["applied_files"] == ["db/schema.sql", "app/api.py", "web/list.js"]
    assert (tmp_path / "web" / "list.js").exists()

    # Every developer sees the shared contract and the stack in its prompt.
    for role in files:
        system = llms[role].prompts[0][0][1]
        assert "GET /items -> 200" in system
        assert "items(id INTEGER PK)" in system
        assert "node - react" in system

    # The reviewer gets the changed files, not the original snapshot.
    assert "FILE: app/api.py" in result["working_file_contents"]
    assert "schema.sql" in result["working_repo_tree"]
    assert result["stage_summary"] == [
        "database_developer done", "backend_developer done",
        "frontend_developer done",
    ]


async def test_no_tasks_is_not_an_error(tmp_path):
    result = await subgraph.develop_node({"workspace": str(tmp_path), "tasks": []})

    assert result["developer_stage"] == "skipped"


async def test_filesystem_is_disconnected_even_when_a_task_fails(
    tmp_path, monkeypatch
):
    created = []

    class Tracking(FakeFilesystem):
        def __init__(self, workspace):
            super().__init__(workspace)
            created.append(self)

    class Boom:
        async def run_task(self, *args, **kwargs):
            raise RuntimeError("blocked")

    monkeypatch.setattr(subgraph, "FilesystemClient", Tracking)
    monkeypatch.setattr(subgraph, "DEVELOPER_ROLES", {"backend_developer": Boom})

    import pytest
    with pytest.raises(RuntimeError, match="blocked"):
        await subgraph.develop_node({
            "workspace": str(tmp_path),
            "tasks": [task("T1", "backend_developer", [])],
        })

    assert created[0].connected is False


async def test_rework_allows_tasks_that_need_no_change(tmp_path, monkeypatch):
    seen = {}

    class Agent:
        async def run_task(self, task, **kwargs):
            seen.update(kwargs)
            return {"summary": "", "applied_files": [], "iterations": 1,
                    "warnings": []}

    monkeypatch.setattr(subgraph, "FilesystemClient", FakeFilesystem)
    monkeypatch.setattr(subgraph, "DEVELOPER_ROLES", {"backend_developer": Agent})

    tasks = [task("T1", "backend_developer", [])]

    await subgraph.develop_node({"workspace": str(tmp_path), "tasks": tasks})
    assert seen["allow_no_changes"] is False

    await subgraph.develop_node({
        "workspace": str(tmp_path), "tasks": tasks, "last_error": "FAILED x",
    })
    assert seen["allow_no_changes"] is True
