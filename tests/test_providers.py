"""Tests for provider model structure — no real API calls."""

from __future__ import annotations

from kognios.models.mistral import MistralModel
from kognios.models.cohere import CohereModel
from kognios.models.ollama import OllamaModel
from kognios.models.xai import XAIModel


def test_mistral_defaults():
    m = MistralModel()
    assert m.model == "mistral-large-latest"
    assert "mistral.ai" in str(m._client.base_url)


def test_mistral_custom_model():
    m = MistralModel(model="mistral-small-latest")
    assert m.model == "mistral-small-latest"


def test_cohere_defaults():
    m = CohereModel()
    assert m.model == "command-a-plus-05-2026"
    assert "cohere.com" in str(m._client.base_url)


def test_cohere_custom_model():
    m = CohereModel(model="command-r-08-2024")
    assert m.model == "command-r-08-2024"


def test_xai_defaults():
    m = XAIModel()
    assert m.model == "grok-4"
    assert "x.ai" in str(m._client.base_url)


def test_xai_custom_model():
    m = XAIModel(model="grok-3")
    assert m.model == "grok-3"


def test_ollama_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.10:11434/v1")
    m = OllamaModel()
    assert "192.168.1.10" in str(m._client.base_url)


# --- Azure OpenAI ---


def test_azure_openai_init():
    from kognios.models.azure import AzureOpenAIModel

    m = AzureOpenAIModel(
        azure_endpoint="https://test.openai.azure.com/",
        api_version="2024-02-01",
        deployment_name="gpt-4o",
        api_key="fake-key",
    )
    assert m.model == "gpt-4o"
    assert isinstance(m._client, __import__("openai").AzureOpenAI)


def test_azure_openai_custom_deployment():
    from kognios.models.azure import AzureOpenAIModel

    m = AzureOpenAIModel(
        azure_endpoint="https://myres.openai.azure.com/",
        deployment_name="gpt-4-turbo",
        api_key="fake-key",
    )
    assert m.model == "gpt-4-turbo"


# --- Vertex AI ---


def test_vertex_ai_import_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "vertexai", None)
    monkeypatch.setitem(sys.modules, "vertexai.generative_models", None)

    import importlib
    import kognios.models.vertex as vertex_mod

    importlib.reload(vertex_mod)

    import pytest

    with pytest.raises(ImportError, match="google-cloud-aiplatform"):
        vertex_mod.VertexAIModel()
