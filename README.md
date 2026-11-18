# CodeCrew

A multi-agent AI software team. You give it a GitHub repository and a request
("add a divide function with tests"); a LangGraph workflow plans the work,
writes the code, runs the tests in a sandbox, reviews the result and waits for
a human to approve it.

## How it works

```
intake -> project manager -> architect -> developers -> tester -> reviewer -> human approval
                                  ^             ^          |                        |
                                  |             +- retry --+ (up to max_attempts)   |
                                  +------------- changes requested ----------------+
```

- **Intake / Project manager / Architect** turn the request into a spec and a
  list of tasks, each assigned to a backend, database or frontend developer.
- **Developers** (`backend/app/graph/developer_subgraph.py`) edit files in a
  cloned workspace through the MCP filesystem server.
- **Tester** runs `pytest` inside a locked-down Docker container
  (`sandbox/`). pytest's exit code decides pass/fail. On failure the output is
  fed back to the developers, up to `max_attempts` (3) times.
- **Reviewer** reviews the change; a human then approves, rejects or requests
  changes (up to `max_revisions`, 2).
- Run state is checkpointed in PostgreSQL so a run can pause for approval and
  resume.

## Requirements

- Python 3.12+
- Node.js (the filesystem MCP server is started with `npx`)
- Docker (test sandbox)
- PostgreSQL (checkpoints)
- A [Groq](https://console.groq.com) API key

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

docker build -t codecrew-sandbox sandbox/
```

Create a `.env` file:

```env
GROQ_API_KEY=your-key
DATABASE_URL=postgresql://user:password@localhost:5432/codecrew
```

Start the API:

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

## Groq settings

The defaults suit Groq's free tier for `openai/gpt-oss-120b`
(8,000 tokens/minute, 200,000 tokens/day). Prompts are trimmed to fit, and the
client waits out per-minute limits.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model name |
| `GROQ_TPM_LIMIT` | `8000` | Tokens per minute allowed by your plan |
| `GROQ_TPM_SAFETY_MARGIN` | `1000` | Headroom below the limit |
| `GROQ_TPD_LIMIT` | `200000` | Daily tokens; a warning is printed at 80 % |
| `GROQ_MAX_TOKENS` | `1200` | Default output tokens per call |
| `GROQ_MAX_RETRIES` | `4` | Retries after an HTTP 429 |
| `GROQ_MAX_RETRY_WAIT` | `120` | A longer `retry-after` (seconds) is treated as an exhausted daily limit and fails fast |

On a paid plan raise `GROQ_TPM_LIMIT`; larger prompt caps are in
`backend/app/agents/developer.py`.

## Tests

```bash
pytest                    # unit tests, no external services needed
pytest -m integration     # needs network, Groq, PostgreSQL and Docker
```

Several files in `backend/tests/` (`run_*.py`, `test_architect.py`,
`test_full_workflow.py`, ...) are manual scripts that call real services; run
them directly with `python backend/tests/<file>.py`.

## Known limitations

- The sandbox has no network and only `pytest` installed, so a target
  repository whose tests need third-party packages (or that has no `tests/`
  folder) will fail the test step. Extend `sandbox/Dockerfile` for the stacks
  you need.
- Large files do not fit the free-tier prompt budget: file contents are cut to
  6,000 characters per developer call.
