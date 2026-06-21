from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints


_REGISTRY: dict[str, Callable] = {}

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
}


def _python_type_to_json(annotation) -> dict:
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return {"type": "array"}
    mapped = _TYPE_MAP.get(annotation)
    if mapped:
        return {"type": mapped}
    return {"type": "string"}


def _build_schema(fn: Callable) -> dict:
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    hints.pop("return", None)

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        annotation = hints.get(name, str)
        prop = _python_type_to_json(annotation)
        prop["description"] = name.replace("_", " ")
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    doc = (fn.__doc__ or "").strip()
    description = doc.split("\n")[0] if doc else fn.__name__

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def tool(fn: Callable) -> Callable:
    fn._tool_schema = _build_schema(fn)
    _REGISTRY[fn.__name__] = fn
    return fn


class ToolRegistry:
    def __init__(self, tools: list[Callable] | None = None):
        self._tools: dict[str, Callable] = {}
        for t in tools or []:
            self.register(t)

    def register(self, fn: Callable) -> None:
        if not hasattr(fn, "_tool_schema"):
            fn._tool_schema = _build_schema(fn)
            fn = tool.__wrapped__(fn) if hasattr(tool, "__wrapped__") else fn
            _REGISTRY[fn.__name__] = fn
        self._tools[fn.__name__] = fn

    def schemas(self) -> list[dict]:
        return [fn._tool_schema for fn in self._tools.values()]

    def call(self, name: str, arguments: dict) -> str:
        fn = self._tools.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = fn(**arguments)
            if asyncio.iscoroutine(result):
                raise TypeError(f"Tool '{name}' is async. Use agent.arun() or registry.acall().")
            return str(result)
        except Exception as exc:
            return f"Error: {exc}"

    async def acall(self, name: str, arguments: dict) -> str:
        """Async-aware tool call — awaits coroutine tools (e.g. MCP tools)."""
        fn = self._tools.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = fn(**arguments)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as exc:
            return f"Error: {exc}"

    def __bool__(self) -> bool:
        return bool(self._tools)
