
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class FilesystemClient:
    def __init__(self, workspace=r"C:\CodeCrew\workspace"):
        self.workspace = workspace
        self.session = None
        self.exit_stack = AsyncExitStack()

        self.server_params = StdioServerParameters(
            command="cmd",
            args=[
                "/c",
                "npx",
                "-y",
                "@modelcontextprotocol/server-filesystem",
                workspace
            ]
        )

    async def connect(self):
        read, write = await self.exit_stack.enter_async_context(
            stdio_client(self.server_params)
        )

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self.session.initialize()
        print("Filesystem MCP Connected!")

    async def disconnect(self):
        await self.exit_stack.aclose()
        self.session = None

    async def call_tool(self, tool_name, arguments):
        if self.session is None:
            raise RuntimeError("Filesystem MCP is not connected")

        result = await self.session.call_tool(tool_name, arguments)

        if result.is_error:
            raise RuntimeError(
                "\n".join(
                    item.text
                    for item in result.content
                    if hasattr(item, "text")
                )
            )

        return "\n".join(
            item.text
            for item in result.content
            if hasattr(item, "text")
        )

    async def list_files(self, path=None):
        return await self.call_tool(
            "list_directory",
            {"path": path or self.workspace}
        )

    async def read_file(self, path):
        return await self.call_tool(
            "read_text_file",
            {"path": path}
        )

    async def read_multiple_files(self, paths):
        return await self.call_tool(
            "read_multiple_files",
            {"paths": paths}
        )

    async def write_file(self, path, content):
        return await self.call_tool(
            "write_file",
            {
                "path": path,
                "content": content
            }
        )

    async def get_directory_tree(self, path=None):
        return await self.call_tool(
            "directory_tree",
            {"path": path or self.workspace}
        )

    async def search_files(self, path, pattern):
        return await self.call_tool(
            "search_files",
            {
                "path": path,
                "pattern": pattern
            }
        )


async def main():
    client = FilesystemClient()

    try:
        await client.connect()

        path = r"C:\CodeCrew\workspace\test.txt"

        await client.write_file(path, "Hello from CodeCrew!")

        content = await client.read_file(path)

        print("File Content:", content)

    finally:
        await client.disconnect()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
