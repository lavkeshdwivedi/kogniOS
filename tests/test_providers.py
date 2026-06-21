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
