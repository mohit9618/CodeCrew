# CodeCrew

A multi-agent AI software team. You give it a GitHub repository and a request
("add an items API with a list page"); a LangGraph workflow plans the work,
writes backend, database and frontend code, runs the project's own tests in a
sandbox, reviews the result, waits for a human to approve it and then opens a
pull request.

## How it works

```
intake -> project manager -> architect -> developers -> tester -> reviewer -> human approval -> publisher
                                  ^            ^          |                          |
                                  |            +- retry --+ (up to max_attempts)     |
                                  +--------------- changes requested ---------------+
```

- **Intake / Project manager / Architect** turn the request into a spec and a
  list of tasks, each assigned to a backend, database or frontend developer.
  The architect also writes a shared **API contract** and **data model**
  (exact endpoints, fields, tables) that every developer receives, so the
  separately written pieces fit together.
- **Developers** (`agents/developer.py`, `graph/developer_subgraph.py`) run
  the tasks in the architect's dependency order. All roles share one
  implementation; `agents/roles.py` gives each its own rules (backend: thin
  handlers, validation, no string-built SQL; database: keys, indexes,
  migrations; frontend: loading/error states, accessibility) plus the
  detected stack and the contract.
- **Tester** installs the project's dependencies and runs tests, lint, type
  check and build in Docker (`sandbox/`). The sandbox result decides
  pass/fail; on failure the output goes back to the developers, up to
  `max_attempts` (3) times.
- **Reviewer** reviews the change; a human approves, rejects or requests
  changes (up to `max_revisions`, 2).
- **Publisher** (`services/publisher.py`) runs after approval: it commits only
  the files the agents changed to a new branch, pushes it and opens a pull
  request.
- Run state is checkpointed in PostgreSQL so a run can pause for approval and
  resume.

### The developer loop

A task is not a single model call. Each developer works in up to three rounds:

1. It sees the task, the repository tree and the files that matter. Files the
   architect named come first, then the best keyword matches
   (`utils/retrieval.py`), cut to the model's token budget.
2. It answers with changes: whole files for new files, or **search/replace
   edits** for existing ones (`utils/edits.py`) so a long file is not
   re-typed. It may also ask to read more files.
3. The changes are applied and syntax-checked. If it asked, the project's
   tests run in the sandbox. Any problem (an edit whose search text was not
   found, a syntax error, a failing test) is shown to it for the next round.

Paths from the model are validated: no `..`, absolute paths, `.git`, `.env`
files or keys.

### The sandbox

`sandbox/stack.py` detects Python (`pyproject.toml`, `requirements*.txt`, ...)
and Node (`package.json`) projects in the repository root and in `backend/`,
`frontend/`, `server/`, `client/`, `api/`, `web/`, `app/`. For each project:

| Step | Network | Python | Node |
| --- | --- | --- | --- |
| install | on | virtualenv in `.codecrew/venv`, `pip install -r ...`, `pip install -e .` | `npm ci` / `npm install` |
| tests (blocking) | off | `pytest` on the test files that were changed, else the whole suite | `npm test` (skipped if it is the `npm init` placeholder) |
| build (blocking) | off | syntax check of changed files | `npm run build` |
| lint, type check (warnings) | off | `ruff check` on changed files | `npm run lint`, `tsc --noEmit` |

The install is skipped when the dependency files have not changed. Containers
run as a non-root user with CPU, memory and process limits and a read-only
root filesystem. Note that the install step runs the target project's own
install scripts with network access, so only point CodeCrew at repositories
you trust.

## Requirements

- Python 3.12+
- Node.js (the filesystem MCP server is started with `npx`)
- Docker (sandbox)
- PostgreSQL (checkpoints)
- A [Groq](https://console.groq.com) API key
- Optional: a GitHub token to open pull requests

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

docker build -t codecrew-sandbox sandbox/
```

Copy `.env.example` to `.env` and fill it in, then start the API:

```bash
uvicorn backend.app.main:app --reload
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/runs/` | Start a run: `{"repo_url": "https://github.com/owner/repo", "request": "..."}` |
| `GET` | `/runs/{run_id}` | Current state and next node |
| `GET` | `/runs/{run_id}/history` | Checkpoint history |
| `POST` | `/runs/{run_id}/resume` | Answer the approval step: `{"decision": "approve" \| "request_changes" \| "reject", "feedback": "..."}` |

After an approval the state contains `publish_status` (`published`,
`skipped` or `failed`), `pr_url` and `publish_error`.

## Models and Groq limits

Every agent role can use its own model, and every model gets its own rate
limiter because Groq limits tokens per minute (TPM) per model. Prompts are
sized from the model's budget, so a model with a larger limit sees more code.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model for every role |
| `GROQ_MODEL_<ROLE>` | | Override for one role: `ARCHITECT`, `REVIEWER`, `TESTER`, `INTAKE`, `PROJECT_MANAGER`, `ANALYZER`, `BACKEND_DEVELOPER`, `DATABASE_DEVELOPER`, `FRONTEND_DEVELOPER` |
| `GROQ_TPM_LIMIT` | per model (below) | Force one TPM budget for all models |
| `GROQ_TPM_SAFETY_MARGIN` | `1000` | Headroom below the limit |
| `GROQ_TPD_LIMIT` | off | Daily tokens; a warning is printed at 80 % |
| `GROQ_MAX_TOKENS` | `1200` | Default output tokens per call |
| `GROQ_MAX_RETRIES` | `4` | Retries after an HTTP 429 or a connection error |
| `GROQ_MAX_RETRY_WAIT` | `120` | A longer `retry-after` (seconds) is treated as an exhausted daily limit and fails fast |

Built-in TPM defaults (`llm/groq_client.py`, `MODEL_TPM`) follow Groq's free
tier at the time of writing: `llama-3.3-70b-versatile` 12,000,
`openai/gpt-oss-120b` and `-20b` 8,000, `moonshotai/kimi-k2-instruct` 10,000,
`qwen/qwen3-32b` 6,000, `meta-llama/llama-4-scout-17b-16e-instruct` 30,000;
unknown models get 6,000. Limits change, so check the rate limits shown for
your account in the Groq console and set `GROQ_TPM_LIMIT` if they differ.

The default is a non-reasoning model on purpose: reasoning models spend part of
the output budget on hidden thinking, which truncates long JSON answers.
`openai/gpt-oss-120b` is a reasonable choice for the architect and reviewer
(`GROQ_MODEL_ARCHITECT=openai/gpt-oss-120b`).

## Publishing

Set `GITHUB_TOKEN` (a token with write access to the repository; a
fine-grained token needs "Contents" and "Pull requests" write permission).
Optionally set `CODECREW_GIT_NAME` and `CODECREW_GIT_EMAIL` for the commit
author. If you do not have push access to the repository you run CodeCrew on,
fork it first. Without a token the run still completes and the changes stay in
the run's workspace.

## Tests

```bash
pytest                    # unit tests, no external services needed
pytest -m integration     # needs network, Groq, PostgreSQL and Docker
```

A few files in `backend/tests/` (`test_architect.py`, `test_full_workflow.py`,
...) are manual scripts that call real services; run them directly with
`python backend/tests/<file>.py`.

## Known limitations

- Node projects are installed with `npm` only (yarn and pnpm lockfiles are
  ignored). Other languages are not detected: a repository with no Python or
  Node project gets no checks, and the tester reports that nothing was run.
- When no test file was changed, the whole test suite runs (limit: 3 minutes).
- On Linux hosts the container user (uid 1000) must be able to write to the
  workspace; Docker Desktop on Windows and macOS needs no setup.
- Free-tier token limits still apply: very large files are shown truncated,
  and every call can wait up to a minute for the rate limit.
