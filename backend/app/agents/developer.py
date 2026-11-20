"""The developer agent: one implementation shared by every developer role.

A task is worked on in a short loop instead of a single shot:

    1. show the model the task, the repository tree and the relevant files
    2. apply the changes it returns (new files, or search/replace edits)
    3. check the result (syntax, optionally the project's tests)
    4. if something is wrong, or the model asked to read more files, show it
       what happened and go round again (at most MAX_ITERATIONS times)

What differs between the backend, database and frontend developers is only
the prompt (agents/roles.py).
"""
import asyncio
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, ValidationError, model_validator

from backend.app.agents.roles import build_system_prompt
from backend.app.llm import get_groq_llm
from backend.app.llm.groq_client import CHARS_PER_TOKEN
from backend.app.utils.edits import EditError, apply_edits
from backend.app.utils.paths import resolve_in_workspace
from backend.app.utils.repo_tree import build_repo_tree
from backend.app.utils.retrieval import find_relevant_files, read_files_text
from backend.app.utils.text import clamp_text, split_budget


load_dotenv()


class Edit(BaseModel):
    search: str = Field(description="Exact existing text, occurs once")
    replace: str = Field(description="Text that replaces it")


class FileChange(BaseModel):
    path: str = Field(description="Relative file path inside workspace")
    content: str | None = Field(
        default=None,
        description="Complete file content (new file or full rewrite)"
    )
    edits: list[Edit] = Field(
        default_factory=list,
        description="Search/replace edits for an existing file"
    )
    summary: str = Field(default="", description="What changed in this file")

    @model_validator(mode="after")
    def content_or_edits(self):
        if (self.content is None) == (not self.edits):
            raise ValueError(
                f"{self.path}: give either `content` or `edits`, not both "
                "and not neither."
            )

        return self


class DeveloperOutput(BaseModel):
    summary: str = Field(default="", description="Overall implementation summary")
    changes: list[FileChange] = Field(default_factory=list)
    read_files: list[str] = Field(
        default_factory=list,
        description="Files the developer wants to see before changing anything"
    )
    run_tests: bool = Field(
        default=False,
        description="Verify the work by running the project's tests"
    )
    questions: list[str] = Field(
        default_factory=list,
        description="Questions or blockers, if any"
    )


# Tokens reserved for the model's answer. Edits are short, so this mostly
# matters for new files.
MAX_OUTPUT_TOKENS = 3_500

# Work rounds per task: change -> check -> fix.
MAX_ITERATIONS = 3

# The model sometimes returns truncated or invalid JSON. Retry a limited
# number of times instead of failing the whole run.
MAX_FORMAT_RETRIES = 1

# estimate_tokens() counts the repr of the message list, which is a little
# larger than the real text (escaped newlines), so leave some room.
PROMPT_SAFETY = 0.85
MIN_PROMPT_CHARS = 2_000

# Share of the prompt space each part may use before leftovers are shared.
PART_WEIGHTS = {
    "task": 0.20,
    "observations": 0.20,
    "tree": 0.15,
    "files": 0.45,
}

MAX_RELEVANT_FILES = 6
MAX_FILES_PER_REQUEST = 5


def is_format_error(error: Exception) -> bool:
    if isinstance(error, (OutputParserException, ValidationError)):
        return True

    # Groq answers HTTP 400 "json_validate_failed" when the model cannot
    # produce valid JSON.
    return (
        type(error).__name__ == "BadRequestError"
        and "json" in str(error).lower()
    )


def find_syntax_errors(workspace: str, paths: list[str]) -> list[str]:
    """Syntax errors in Python files, found without running anything."""
    errors = []

    for path in paths:
        if not path.endswith(".py"):
            continue

        try:
            source = resolve_in_workspace(workspace, path).read_text(
                encoding="utf-8"
            )
            compile(source, path, "exec")
        except SyntaxError as error:
            errors.append(f"{path}:{error.lineno}: {error.msg}")
        except (OSError, ValueError):
            continue

    return errors


class DeveloperAgent:
    def __init__(self, specialization="general", llm=None):
        self.specialization = specialization

        self.llm = llm or get_groq_llm(
            max_tokens=MAX_OUTPUT_TOKENS,
            role=f"{specialization}_developer"
        )

        self.structured_llm = self.llm.with_structured_output(
            DeveloperOutput,
            method="json_mode"
        )

    async def generate_changes(
        self,
        task: str,
        repo_tree: str,
        file_contents: str,
        observations: str = "",
        stack_summary: str = "",
        api_contract: list[str] | None = None,
        data_model: list[str] | None = None,
    ) -> DeveloperOutput:
        system = build_system_prompt(
            self.specialization, stack_summary, api_contract, data_model
        )

        available = int(
            self.llm.prompt_char_budget(
                overhead_tokens=len(system) // CHARS_PER_TOKEN + 100
            ) * PROMPT_SAFETY
        )

        limits = split_budget(
            max(available, MIN_PROMPT_CHARS),
            {
                "task": len(task),
                "observations": len(observations),
                "tree": len(repo_tree),
                "files": len(file_contents),
            },
            PART_WEIGHTS
        )

        task = clamp_text(task, limits["task"], "task")
        # Errors and test summaries come last, so keep the end.
        observations = clamp_text(
            observations, limits["observations"], "observations", keep="tail"
        )
        repo_tree = clamp_text(repo_tree, limits["tree"], "repository tree")
        file_contents = clamp_text(
            file_contents, limits["files"], "file contents"
        )

        human = f"DEVELOPMENT TASK:\n{task}\n\n"

        if observations:
            human += f"OBSERVATIONS FROM YOUR PREVIOUS STEP:\n{observations}\n\n"

        human += (
            f"REPOSITORY DIRECTORY TREE:\n{repo_tree}\n\n"
            f"FILE CONTENTS (current state on disk):\n{file_contents}\n\n"
            "Implement only this task."
        )

        messages = [("system", system), ("human", human)]

        for attempt in range(MAX_FORMAT_RETRIES + 1):
            try:
                return await self.structured_llm.ainvoke(messages)

            except Exception as error:
                if (
                    not is_format_error(error)
                    or attempt == MAX_FORMAT_RETRIES
                ):
                    raise

                print(
                    "[Developer] Invalid or truncated JSON from the model; "
                    f"retrying ({attempt + 1}/{MAX_FORMAT_RETRIES})."
                )

                messages = messages + [
                    (
                        "human",
                        "Your previous reply was not valid JSON, was cut "
                        "off, or broke the format rules. Reply again with "
                        "ONE complete, valid JSON object. Prefer small "
                        "edits over rewriting whole files."
                    )
                ]

    async def apply_changes(
        self,
        output: DeveloperOutput,
        filesystem
    ) -> tuple[list[str], list[str]]:
        """Write the changes to the workspace.

        Returns (applied paths, error messages). A change that cannot be
        applied (bad path, search text not found) is reported back to the
        model instead of aborting the run.
        """
        workspace = filesystem.workspace
        applied = []
        errors = []

        for change in output.changes:
            try:
                target = resolve_in_workspace(workspace, change.path)

                if change.content is not None:
                    new_content = change.content

                else:
                    if not target.is_file():
                        raise EditError(
                            f"{change.path} does not exist, so it cannot "
                            "be edited. Create it with `content`."
                        )

                    current = target.read_bytes().decode("utf-8")
                    new_content = apply_edits(
                        current, change.edits, change.path
                    )

            except (ValueError, UnicodeDecodeError) as error:
                errors.append(str(error))
                continue

            await filesystem.create_directory(str(target.parent))
            await filesystem.write_file(str(target), new_content)

            if change.path not in applied:
                applied.append(change.path)

        return applied, errors

    async def run_task(
        self,
        task: dict,
        *,
        task_text: str,
        workspace: str,
        filesystem,
        stack_summary: str = "",
        api_contract: list[str] | None = None,
        data_model: list[str] | None = None,
        check_runner: Callable[[str, list[str]], dict] | None = None,
        allow_no_changes: bool = False,
        applied_before: list[str] | None = None,
    ) -> dict:
        """Work on one task until it is done or MAX_ITERATIONS is reached.

        `check_runner(workspace, changed_files)` runs the project's tests in
        the sandbox; it is only called when the model asks for it.
        Returns {"summary", "applied_files", "iterations", "warnings"}.
        """
        task_id = task.get("task_id", "?")
        explicit = list(dict.fromkeys(
            task.get("files_to_modify", []) + task.get("files_to_read", [])
        ))

        query = " ".join([
            task.get("title", ""),
            task.get("description", ""),
            *task.get("acceptance_criteria", []),
        ])

        relevant = await asyncio.to_thread(
            find_relevant_files,
            workspace, query, explicit, MAX_RELEVANT_FILES
        )

        requested: list[str] = []
        observations = ""
        last_problems: list[str] = []
        applied: list[str] = []
        summaries: list[str] = []
        warnings: list[str] = []

        # Characters of file content worth preparing; generate_changes
        # trims further to what the model's token budget allows.
        file_char_budget = int(
            self.llm.prompt_char_budget() * PROMPT_SAFETY
            * PART_WEIGHTS["files"] * 2
        )

        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1
            is_last = iteration == MAX_ITERATIONS

            # Files changed in this task come first (their current state is
            # what a fix needs), then files the model asked for.
            ordered = list(dict.fromkeys(applied + requested + relevant))

            file_contents = await asyncio.to_thread(
                read_files_text, workspace, ordered, file_char_budget
            )

            repo_tree = await asyncio.to_thread(build_repo_tree, workspace)

            print(f"[Developer] task {task_id}: round {iteration}/{MAX_ITERATIONS}")

            output = await self.generate_changes(
                task=task_text,
                repo_tree=repo_tree,
                file_contents=file_contents,
                observations=observations,
                stack_summary=stack_summary,
                api_contract=api_contract,
                data_model=data_model,
            )

            if output.questions:
                raise RuntimeError(
                    f"Task {task_id} has blockers: {output.questions}"
                )

            if output.summary:
                summaries.append(output.summary)

            # `problems` must be fixed; `info` just tells the model what
            # happened (e.g. the files it asked for are now shown).
            problems: list[str] = []
            info: list[str] = []

            new_applied, apply_errors = await self.apply_changes(
                output, filesystem
            )

            for path in new_applied:
                if path not in applied:
                    applied.append(path)

            problems += [
                f"Could not apply a change: {error}"
                for error in apply_errors
            ]

            wanted = [
                path for path in output.read_files
                if path not in requested
            ][:MAX_FILES_PER_REQUEST]

            if wanted:
                requested += wanted
                info.append(
                    "The files you asked to read are now shown: "
                    + ", ".join(wanted)
                )

            if not output.changes and not wanted:
                # The model answered the last round's problems with no
                # change at all. Keep them as warnings: they are unresolved.
                warnings += last_problems

                if applied or allow_no_changes:
                    break

                raise RuntimeError(
                    f"Task {task_id} generated no changes."
                )

            problems += [
                f"Syntax error: {error}"
                for error in find_syntax_errors(workspace, new_applied)
            ]

            if output.run_tests and check_runner and applied and not problems:
                result = await asyncio.to_thread(
                    check_runner,
                    workspace,
                    applied + list(applied_before or [])
                )

                if not result.get("passed"):
                    problems.append(
                        "Running the project's checks failed:\n"
                        + clamp_text(
                            result.get("stdout", ""), 3_000, "check output",
                            keep="tail"
                        )
                    )

            # Another round is only worth it to fix a problem, or to show
            # files the model asked for before it has changed anything. If
            # it changed files cleanly and merely asked to read more, it is
            # done: a further round would just re-apply the same edits.
            if not problems and not (wanted and not output.changes):
                break

            last_problems = problems
            observations = "\n".join(problems + info)

            if is_last:
                warnings += problems
                break

        if not applied and not allow_no_changes:
            raise RuntimeError(
                f"Task {task_id} produced no usable changes. "
                + " ".join(warnings or [observations])
            )

        return {
            "summary": "\n".join(dict.fromkeys(summaries)),
            "applied_files": applied,
            "iterations": iteration,
            "warnings": warnings,
        }
