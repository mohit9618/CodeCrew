import asyncio

from backend.app.agents import developer as developer_module
from backend.app.agents.developer import (
    MAX_FILE_CONTENT_CHARS,
    MAX_TASK_CHARS,
    MAX_TREE_CHARS,
    DeveloperAgent,
    DeveloperOutput,
)
from backend.app.llm.groq_client import GroqRateLimiter
from backend.app.utils.repo_tree import build_repo_tree


def make_workspace(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("print('hi')")
    (tmp_path / "README.md").write_text("# demo")

    for ignored in (".git", "node_modules", "venv", "__pycache__"):
        (tmp_path / ignored).mkdir()
        for i in range(50):
            (tmp_path / ignored / f"junk_{i}.js").write_text("x")

    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")

    return tmp_path


def test_repo_tree_skips_ignored_and_sensitive_files(tmp_path):
    tree = build_repo_tree(str(make_workspace(tmp_path)))

    assert "main.py" in tree
    assert "README.md" in tree

    for unwanted in (".git", "node_modules", "venv", "__pycache__",
                     "junk_", ".env", "logo.png"):
        assert unwanted not in tree


def test_repo_tree_is_capped(tmp_path):
    for i in range(500):
        (tmp_path / f"module_{i}.py").write_text("x")

    tree = build_repo_tree(str(tmp_path), max_chars=500)

    assert len(tree) < 600
    assert tree.endswith("(tree truncated)")


class _FakeStructuredLLM:
    def __init__(self):
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return DeveloperOutput(summary="ok", changes=[])


def test_developer_prompt_fits_default_tpm_budget():
    agent = DeveloperAgent.__new__(DeveloperAgent)
    agent.specialization = "backend"
    agent.structured_llm = _FakeStructuredLLM()

    asyncio.run(
        agent.generate_changes(
            task="t" * 50_000,
            repo_tree="r" * 50_000,
            file_contents="f" * 50_000
        )
    )

    human = agent.structured_llm.messages[1][1]

    assert len(human) < (
        MAX_TASK_CHARS + MAX_TREE_CHARS + MAX_FILE_CONTENT_CHARS + 1_000
    )

    # Same estimate SharedGroqClient uses: chars // 4 + reserved output.
    estimate = len(str(agent.structured_llm.messages)) // 4 + 2_500
    limiter = GroqRateLimiter(tpm_limit=8000, safety_margin=1000)

    assert estimate <= limiter.max_budget


def test_clamp_text_leaves_short_text_alone():
    assert developer_module.clamp_text("abc", 10, "x") == "abc"
