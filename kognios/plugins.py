# kognios/plugins.py
"""
Plugin discovery via Python entry points.

Third-party packages register plugins in their pyproject.toml:

    [project.entry-points."kognios.providers"]
    my_provider = "my_package:MyModel"

    [project.entry-points."kognios.tools"]
    my_tool = "my_package:my_tool_fn"

    [project.entry-points."kognios.memory"]
    redis = "kognios_redis:RedisMemory"

Then load them at runtime:
    from kognios.plugins import load_plugins
    providers = load_plugins("kognios.providers")
"""

from __future__ import annotations

import importlib  # noqa: F401
from importlib.metadata import entry_points


_GROUPS = ("kognios.providers", "kognios.tools", "kognios.memory", "kognios.knowledge")


def load_plugins(group: str) -> dict[str, object]:
    """Discover and load all registered plugins for a given entry-point group.

    Returns a dict of {name: loaded_object} for each registered entry point.
    If an entry point fails to load, it is skipped and a warning is printed.
    """
    loaded: dict[str, object] = {}
    eps = entry_points(group=group)
    for ep in eps:
        try:
            loaded[ep.name] = ep.load()
        except Exception as exc:
            print(
                f"[kognios] Warning: failed to load plugin '{ep.name}' from group '{group}': {exc}"
            )
    return loaded


def discover_all() -> dict[str, dict[str, object]]:
    """Load all known kognios plugin groups.

    Returns a nested dict: {group_name: {plugin_name: plugin_object}}.
    """
    return {group: load_plugins(group) for group in _GROUPS}


def list_plugins(group: str) -> list[str]:
    """Return the names of all registered plugins for a group (without loading them)."""
    return [ep.name for ep in entry_points(group=group)]
