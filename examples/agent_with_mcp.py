"""
MCP example — agent backed by an MCP filesystem server.

Install the server first:
    pip install 'kognios[mcp]'
    uvx install mcp-server-filesystem   # or: npm install -g @modelcontextprotocol/server-filesystem

Then run:
    python examples/agent_with_mcp.py
"""

import asyncio

from kognios import Agent, AnthropicModel, MCPClient


async def async_example() -> None:
    # Connect to a local filesystem MCP server (reads the current directory)
    async with MCPClient.stdio("uvx", ["mcp-server-filesystem", "."]) as mcp:
        print("Available MCP tools:", mcp.tool_names())

        agent = Agent(
            model=AnthropicModel(),
            tools=mcp.tools(),
            instructions="You are a helpful assistant with access to the local filesystem.",
        )

        result = await agent.arun("List the Python files in the current directory.")
        print(result)


def sync_example() -> None:
    # Same thing, but using the synchronous API
    with MCPClient.stdio("uvx", ["mcp-server-filesystem", "."]).sync() as mcp:
        agent = Agent(
            model=AnthropicModel(),
            tools=mcp.tools(),
            instructions="You are a helpful assistant with access to the local filesystem.",
        )

        result = agent.run("How many Python files are there?")
        print(result)


if __name__ == "__main__":
    asyncio.run(async_example())
    # sync_example()  # uncomment to test the sync path
