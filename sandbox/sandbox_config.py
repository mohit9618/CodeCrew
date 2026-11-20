import os

SANDBOX_IMAGE = os.getenv("CODECREW_SANDBOX_IMAGE", "codecrew-sandbox")

CPU_LIMIT = float(os.getenv("CODECREW_SANDBOX_CPUS", "2.0"))
MEMORY_LIMIT = os.getenv("CODECREW_SANDBOX_MEMORY", "1g")
PIDS_LIMIT = int(os.getenv("CODECREW_SANDBOX_PIDS", "256"))
TMPFS_SIZE = os.getenv("CODECREW_SANDBOX_TMPFS", "512m")

# Longest output kept per check. The tail is kept: pytest and most tools
# print their summary last.
MAX_OUTPUT_CHARS = 6_000
