"""Tests for browse tool and BedrockModel structure — no real network/AWS calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_browse_falls_back_to_urllib():
    """browse() should work without playwright by using urllib fallback."""
    import sys
    import types

    fake_html = b"<html><body><p>Hello browser world</p></body></html>"

    class FakeResponse:
        def read(self):
            return fake_html

        def decode(self, *a, **kw):
            return fake_html.decode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    # Remove any cached playwright import so the ImportError path is exercised
    playwright_modules = [k for k in sys.modules if k.startswith("playwright")]
    saved = {k: sys.modules.pop(k) for k in playwright_modules}

    # Inject a fake playwright module that raises ImportError on import of sync_playwright
    fake_playwright_pkg = types.ModuleType("playwright")
    fake_sync_api = types.ModuleType("playwright.sync_api")

    def _raise(*a, **kw):
        raise ImportError("playwright not available")

    fake_sync_api.sync_playwright = _raise  # type: ignore[attr-defined]
    sys.modules["playwright"] = fake_playwright_pkg
    sys.modules["playwright.sync_api"] = fake_sync_api

    try:
        # Re-import browse with the fake playwright in place
        import importlib
        import kognios.tools.builtins.browser as _browser_mod

        importlib.reload(_browser_mod)
        browse = _browser_mod.browse

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = browse("http://example.com")
        assert "Hello browser world" in result
    finally:
        # Restore original playwright modules
        for k in ("playwright", "playwright.sync_api"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def test_bedrock_model_init():
    """BedrockModel instantiates correctly (no real AWS connection)."""
    import anthropic

    with patch.object(anthropic, "AnthropicBedrock", return_value=MagicMock()):
        from kognios.models.bedrock import BedrockModel

        m = BedrockModel(
            aws_access_key="fake_key",
            aws_secret_key="fake_secret",
            aws_region="us-east-1",
        )
        assert m.model == "anthropic.claude-sonnet-5"


def test_bedrock_custom_model():
    import anthropic

    with patch.object(anthropic, "AnthropicBedrock", return_value=MagicMock()):
        from kognios.models.bedrock import BedrockModel

        m = BedrockModel(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            aws_access_key="k",
            aws_secret_key="s",
        )
        assert "haiku" in m.model
