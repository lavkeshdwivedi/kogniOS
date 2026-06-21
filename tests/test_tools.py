from kognios.tools.registry import tool, ToolRegistry


def test_tool_schema_generation():
    @tool
    def add(x: int, y: int) -> int:
        """Add two integers."""
        return x + y

    schema = add._tool_schema
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "add"
    assert fn["description"] == "Add two integers."
    props = fn["parameters"]["properties"]
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "integer"
    assert set(fn["parameters"]["required"]) == {"x", "y"}


def test_tool_optional_param_not_required():
    @tool
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}"

    schema = greet._tool_schema
    required = schema["function"]["parameters"]["required"]
    assert "name" in required
    assert "greeting" not in required


def test_registry_call():
    @tool
    def double(n: int) -> int:
        """Double a number."""
        return n * 2

    registry = ToolRegistry([double])
    result = registry.call("double", {"n": 5})
    assert result == "10"


def test_registry_unknown_tool():
    registry = ToolRegistry([])
    result = registry.call("nonexistent", {})
    assert "unknown tool" in result


def test_registry_tool_error():
    @tool
    def fail(x: int) -> int:
        """Always fails."""
        raise ValueError("boom")

    registry = ToolRegistry([fail])
    result = registry.call("fail", {"x": 1})
    assert "Error" in result


def test_registry_schemas_returns_list():
    @tool
    def noop() -> str:
        """Does nothing."""
        return ""

    registry = ToolRegistry([noop])
    schemas = registry.schemas()
    assert isinstance(schemas, list)
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
