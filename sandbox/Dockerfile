
FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir pytest

RUN useradd -m runner

USER runner

CMD ["python", "--version"]
