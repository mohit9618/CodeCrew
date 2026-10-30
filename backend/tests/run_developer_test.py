
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.developer import DeveloperAgent
from app.mcp.clients.filesystem import FilesystemClient


async def main():
    fs = FilesystemClient()

    try:
        await fs.connect()

        repo_tree = await fs.get_directory_tree()
        file_contents = await fs.read_file(
            r"C:\CodeCrew\workspace\calculator.py"
        )

        agent = DeveloperAgent()

        result = await agent.generate_changes(
            task="Add a subtract function to calculator.py.",
            repo_tree=repo_tree,
            file_contents=file_contents
        )

        print("\nDEVELOPER AGENT OUTPUT\n")
        print(result.model_dump_json(indent=2))

        applied_files = await agent.apply_changes(result, fs)

        print("\nAPPLIED FILES:")
        for file in applied_files:
            print(file)

        updated_content = await fs.read_file(
            r"C:\CodeCrew\workspace\calculator.py"
        )

        print("\nUPDATED CALCULATOR.PY:")
        print(updated_content)

    finally:
        await fs.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
