try:
    from importlib.metadata import version

    __version__ = version("kognios")
except Exception:
    __version__ = "unknown"

from .agent import Agent
from .team import Team
from .tools.registry import tool, ToolRegistry
from .memory.short_term import ShortTermMemory
from .memory.long_term import LongTermMemory
from .knowledge.sqlite_fts import SQLiteKnowledge
from .knowledge.numpy_vector import NumpyVectorKnowledge
from .models.anthropic import AnthropicModel
from .models.openai import OpenAIModel
from .models.groq import GroqModel
from .models.gemini import GeminiModel
from .models.ollama import OllamaModel
from .models.mistral import MistralModel
from .models.cohere import CohereModel
from .models.bedrock import BedrockModel
from .models.xai import XAIModel
from .models.chain import ModelChain, free_tier_chain
from .models.together import TogetherModel
from .mcp import MCPClient
from .eval import AgentEvaluator, EvalCase, EvalReport
from .guardrails import GuardrailError, block_keywords, max_length, pii_scrubber
from .tracing import Tracer
from .plugins import load_plugins, discover_all, list_plugins

__all__ = [
    "Agent",
    "Team",
    "tool",
    "ToolRegistry",
    "ShortTermMemory",
    "LongTermMemory",
    "SQLiteKnowledge",
    "NumpyVectorKnowledge",
    "AnthropicModel",
    "OpenAIModel",
    "GroqModel",
    "GeminiModel",
    "OllamaModel",
    "MistralModel",
    "CohereModel",
    "BedrockModel",
    "XAIModel",
    "ModelChain",
    "free_tier_chain",
    "TogetherModel",
    "MCPClient",
    "AgentEvaluator",
    "EvalCase",
    "EvalReport",
    "GuardrailError",
    "block_keywords",
    "max_length",
    "pii_scrubber",
    "Tracer",
    "load_plugins",
    "discover_all",
    "list_plugins",
]
