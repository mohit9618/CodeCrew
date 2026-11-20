import asyncio

from backend.app.agents import developer as developer_module
import pytest

from backend.app.agents.developer import (
    MAX_OUTPUT_TOKENS,
    DeveloperAgent,
    DeveloperOutput,
)
from backend.app.llm.groq_client import GroqRateLimiter, SharedGroqClient
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


@pytest.mark.parametrize("tpm_limit", [6_000, 8_000, 12_000])
def test_developer_prompt_fits_the_models_tpm_budget(tpm_limit):
    """Whatever the model's limit, the prompt must stay under it.

    Prompt size is derived from the model's budget, so a small model gets a
    smaller prompt instead of a ValueError from the rate limiter.
    """
    limiter = GroqRateLimiter(tpm_limit=tpm_limit, safety_margin=1000)
    client = SharedGroqClient(
        limiter=limiter,
        model="test-model",
        api_key="test-key",
        max_tokens=MAX_OUTPUT_TOKENS
    )

    agent = DeveloperAgent.__new__(DeveloperAgent)
    agent.specialization = "backend"
    agent.llm = client
    agent.structured_llm = _FakeStructuredLLM()

    asyncio.run(
        agent.generate_changes(
            task="t" * 50_000,
            repo_tree="r" * 50_000,
            file_contents="f" * 50_000,
            observations="o" * 50_000
        )
    )

    # Same estimate SharedGroqClient uses for the rate limiter.
    estimate = client.estimate_tokens(agent.structured_llm.messages)

    assert estimate <= limiter.max_budget


def test_clamp_text_leaves_short_text_alone():
    assert developer_module.clamp_text("abc", 10, "x") == "abc"
