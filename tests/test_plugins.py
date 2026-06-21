"""Tests for the plugin discovery system."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
from importlib.metadata import EntryPoint


def _make_ep(name: str, value: str, group: str) -> MagicMock:
    ep = MagicMock(spec=EntryPoint)
    ep.name = name
    ep.group = group
    ep.load = MagicMock(return_value=f"loaded:{value}")
    return ep


def test_load_plugins_returns_loaded_objects():
    from kognios.plugins import load_plugins

    ep1 = _make_ep("provider_a", "PackageA:ModelA", "kognios.providers")
    with patch("kognios.plugins.entry_points", return_value=[ep1]):
        result = load_plugins("kognios.providers")
    assert result == {"provider_a": "loaded:PackageA:ModelA"}


def test_load_plugins_skips_failures(capsys):
    from kognios.plugins import load_plugins

    ep = _make_ep("bad_plugin", "bad:Bad", "kognios.providers")
    ep.load = MagicMock(side_effect=ImportError("missing dep"))
    with patch("kognios.plugins.entry_points", return_value=[ep]):
        result = load_plugins("kognios.providers")
    assert result == {}
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_list_plugins_returns_names():
    from kognios.plugins import list_plugins

    ep1 = _make_ep("tool_x", "pkg:fn", "kognios.tools")
    ep2 = _make_ep("tool_y", "pkg:fn2", "kognios.tools")
    with patch("kognios.plugins.entry_points", return_value=[ep1, ep2]):
        names = list_plugins("kognios.tools")
    assert set(names) == {"tool_x", "tool_y"}


def test_discover_all_covers_all_groups():
    from kognios.plugins import discover_all, _GROUPS

    with patch("kognios.plugins.load_plugins", return_value={}) as mock_load:
        result = discover_all()
    assert set(result.keys()) == set(_GROUPS)
    assert mock_load.call_count == len(_GROUPS)


def test_version_is_set():
    import kognios

    assert hasattr(kognios, "__version__")
    assert kognios.__version__ != ""
