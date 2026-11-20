# Sandbox image used to install and check the projects CodeCrew edits.
#
# Build once:  docker build -t codecrew-sandbox sandbox/
#
# It carries both runtimes (Python and Node) plus the common tools, so the
# per-run containers only need to install the target project's own
# dependencies. Those are installed with network access; tests, lint and
# type checks run with the network disabled (see sandbox/execution.py).
FROM python:3.12-slim

# Node 22 LTS, copied from the official image: Debian's own nodejs package is
# too old for current frameworks (Vite, Next.js, ...).
COPY --from=node:22-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

RUN pip install --no-cache-dir pytest ruff

RUN useradd -m -u 1000 runner

USER runner

WORKDIR /workspace

CMD ["python", "--version"]
