"""Tests for MCPClient — uses mocks so no real MCP server is required."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kognios.mcp.client import MCPClient, _extract_text, _to_kognios_schema


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_tool(name: str, description: str = "", schema: dict | None = None):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema or {"type": "object", "properties": {"q": {"type": "string"}}}
    return t


def _make_text_content(text: str):
    c = MagicMock()
    c.text = text
    del c.data
    return c


def _make_call_result(text: str):
    r = MagicMock()
    r.content = [_make_text_content(text)]
    return r


def _make_list_result(tools):
    r = MagicMock()
    r.tools = tools
    return r


# ── schema helpers ────────────────────────────────────────────────────────────


def test_to_kognios_schema():
    t = _make_tool("search", "Search the web", {"type": "object", "properties": {}})
    schema = _to_kognios_schema(t)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert schema["function"]["description"] == "Search the web"


def test_extract_text_from_text_content():
    result = _make_call_result("hello world")
    assert _extract_text(result) == "hello world"


def test_extract_text_multiple_parts():
    c1 = _make_text_content("part one")
    c2 = _make_text_content("part two")
    r = MagicMock()
    r.content = [c1, c2]
    assert _extract_text(r) == "part one\npart two"


# ── async connect / tools ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_tools_list():
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(
        return_value=_make_list_result(
            [_make_tool("read_file", "Read a file"), _make_tool("write_file", "Write a file")]
        )
    )
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=False)

    mock_transport_cm = AsyncMock()
    mock_transport_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
    mock_transport_cm.__aexit__ = AsyncMock(return_value=False)

    client = MCPClient.stdio("fake-cmd", ["--arg"])

    with (
        patch("mcp.client.stdio.stdio_client", return_value=mock_transport_cm),
        patch("mcp.ClientSession", return_value=mock_session_cm),
    ):
        async with client:
            tools = client.tools()
            assert len(tools) == 2
            assert tools[0].__name__ == "read_file"
            assert tools[1].__name__ == "write_file"
            assert hasattr(tools[0], "_tool_schema")


@pytest.mark.asyncio
async def test_async_tool_call():
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(
        return_value=_make_list_result([_make_tool("echo", "Echo input")])
    )
    mock_session.call_tool = AsyncMock(return_value=_make_call_result("hello from MCP"))

    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=False)

    mock_transport_cm = AsyncMock()
    mock_transport_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
    mock_transport_cm.__aexit__ = AsyncMock(return_value=False)

    client = MCPClient.stdio("fake-cmd")

    with (
        patch("mcp.client.stdio.stdio_client", return_value=mock_transport_cm),
        patch("mcp.ClientSession", return_value=mock_session_cm),
    ):
        async with client:
            tool_fn = client.tools()[0]
            result = await tool_fn(q="test")
            assert result == "hello from MCP"
            mock_session.call_tool.assert_awaited_once_with("echo", {"q": "test"})


@pytest.mark.asyncio
async def test_tools_raises_when_not_connected():
    client = MCPClient.stdio("fake-cmd")
    with pytest.raises(RuntimeError, match="not connected"):
        client.tools()


# ── ToolRegistry acall ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registry_acall_awaits_async_tool():
    from kognios.tools.registry import ToolRegistry

    async def async_tool(x: str) -> str:
        """An async tool."""
        return f"async:{x}"

    async_tool._tool_schema = {
        "type": "function",
        "function": {
            "name": "async_tool",
            "description": "An async tool.",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    }

    registry = ToolRegistry([async_tool])
    result = await registry.acall("async_tool", {"x": "hello"})
    assert result == "async:hello"


@pytest.mark.asyncio
async def test_registry_acall_works_with_sync_tool():
    from kognios.tools.registry import ToolRegistry

    def sync_tool(x: str) -> str:
        """A sync tool."""
        return f"sync:{x}"

    registry = ToolRegistry([sync_tool])
    result = await registry.acall("sync_tool", {"x": "world"})
    assert result == "sync:world"


# ── schema round-trip ─────────────────────────────────────────────────────────


def test_tool_schema_shape():
    t = _make_tool(
        "list_dir",
        "List directory contents",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    schema = _to_kognios_schema(t)
    assert schema["function"]["parameters"]["required"] == ["path"]
