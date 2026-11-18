import os

from dotenv import load_dotenv

# Load the real .env first so dummy values never shadow real ones.
load_dotenv()

# Importing the app must not need real services. Pool/LLM objects are
# created lazily or with open=False, so placeholders are enough for the
# unit tests. Tests that really need Groq/Postgres are marked
# "integration" and skipped by default.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/codecrew"
)
os.environ.setdefault("GROQ_API_KEY", "test-key")
