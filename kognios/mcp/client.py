from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


class MCPClient:
    """
    Connect to an MCP server and expose its tools to a KogniOS agent.

    Supports stdio (local process) and SSE (HTTP) transports.

    Async usage (recommended with ``agent.arun``)::

        async with MCPClient.stdio("uvx", ["mcp-server-git", "--repository", "."]) as mcp:
            agent = Agent(model=AnthropicModel(), tools=mcp.tools())
            print(await agent.arun("Summarise the last 5 commits"))

    Sync usage (with ``agent.run``)::

        with MCPClient.stdio("uvx", ["mcp-server-git", "--repository", "."]).sync() as mcp:
            agent = Agent(model=AnthropicModel(), tools=mcp.tools())
            print(agent.run("Summarise the last 5 commits"))

    Multiple servers::

        async with MCPClient.stdio("uvx", ["mcp-server-filesystem", "."]) as fs, \\
                   MCPClient.sse("http://localhost:8000/sse") as remote:
            agent = Agent(model=..., tools=[*fs.tools(), *remote.tools()])
    """

    def __init__(self, transport: str, **transport_kwargs: Any) -> None:
        self._transport = transport
        self._transport_kwargs = transport_kwargs
        self._session: Any = None
        self._transport_cm: Any = None
        self._session_cm: Any = None
        self._tool_schemas: list[dict] = []
        self._tool_callables: list[Callable] = []

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> MCPClient:
        """Launch a local MCP server process over stdin/stdout."""
        return cls("stdio", command=command, args=args or [], env=env)

    @classmethod
    def sse(cls, url: str) -> MCPClient:
        """Connect to a remote MCP server over Server-Sent Events."""
        return cls("sse", url=url)

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> MCPClient:
        await self._connect_async()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._disconnect_async()

    async def _connect_async(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
        except ImportError:
            raise ImportError(
                "MCP support requires the 'mcp' package. "
                "Install it with: pip install 'kognios[mcp]'"
            )

        if self._transport == "stdio":
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self._transport_kwargs["command"],
                args=self._transport_kwargs.get("args", []),
                env=self._transport_kwargs.get("env"),
            )
            self._transport_cm = stdio_client(params)
        elif self._transport == "sse":
            from mcp.client.sse import sse_client

            self._transport_cm = sse_client(self._transport_kwargs["url"])
        else:
            raise ValueError(f"Unsupported transport: {self._transport!r}")

        read, write = await self._transport_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        await self._load_tools_async()

    async def _disconnect_async(self) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
        if self._transport_cm is not None:
            await self._transport_cm.__aexit__(None, None, None)

    async def _load_tools_async(self) -> None:
        result = await self._session.list_tools()
        self._tool_schemas = []
        self._tool_callables = []
        for t in result.tools:
            schema = _to_kognios_schema(t)
            self._tool_schemas.append(schema)
            self._tool_callables.append(self._make_async_tool(t.name, schema))

    def _make_async_tool(self, name: str, schema: dict) -> Callable:
        session = self._session

        async def fn(**kwargs: Any) -> str:
            result = await session.call_tool(name, kwargs)
            return _extract_text(result)

        fn.__name__ = name  # type: ignore[attr-defined]
        fn.__doc__ = schema["function"].get("description", name)  # type: ignore[attr-defined]
        fn._tool_schema = schema  # type: ignore[attr-defined]
        return fn  # type: ignore[return-value]

    # ── Sync wrapper ──────────────────────────────────────────────────────────

    @contextmanager
    def sync(self):
        """
        Sync context manager — runs the MCP connection in a background thread.

        The tools returned by ``mcp.tools()`` are synchronous callables
        suitable for ``agent.run()`` and ``agent.stream()``.
        """
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            _run_sync(self._connect_async(), loop)
            self._patch_tools_for_sync(loop)
            yield self
        finally:
            _run_sync(self._disconnect_async(), loop)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)

    def _patch_tools_for_sync(self, loop: asyncio.AbstractEventLoop) -> None:
        session = self._session
        new_callables: list[Callable] = []
        for schema in self._tool_schemas:
            name = schema["function"]["name"]

            async def _call(n: str = name, **kwargs: Any) -> str:
                result = await session.call_tool(n, kwargs)
                return _extract_text(result)

            def fn(
                _loop: asyncio.AbstractEventLoop = loop,
                _c: Any = _call,
                **kwargs: Any,
            ) -> str:
                future = asyncio.run_coroutine_threadsafe(_c(**kwargs), _loop)
                return future.result(timeout=60)

            fn.__name__ = name  # type: ignore[attr-defined]
            fn.__doc__ = schema["function"].get("description", name)  # type: ignore[attr-defined]
            fn._tool_schema = schema  # type: ignore[attr-defined]
            new_callables.append(fn)  # type: ignore[arg-type]
        self._tool_callables = new_callables

    # ── Public API ────────────────────────────────────────────────────────────

    def tools(self) -> list[Callable]:
        """Return tool callables for ``Agent(tools=mcp.tools())``."""
        if not self._tool_callables:
            raise RuntimeError(
                "MCPClient is not connected. "
                "Use 'async with MCPClient...' or 'MCPClient(...).sync()' first."
            )
        return list(self._tool_callables)

    def tool_names(self) -> list[str]:
        """Return the names of available tools."""
        return [s["function"]["name"] for s in self._tool_schemas]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_kognios_schema(tool: Any) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or tool.name,
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def _extract_text(result: Any) -> str:
    parts: list[str] = []
    for item in getattr(result, "content", []):
        if hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "data"):
            parts.append(str(item.data))
    return "\n".join(parts) if parts else repr(result)


def _run_sync(coro: Any, loop: asyncio.AbstractEventLoop) -> Any:
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)
