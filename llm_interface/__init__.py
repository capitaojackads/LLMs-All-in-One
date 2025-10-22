"""
LLMs All-in-One: A unified interface for all major LLM providers.

This package provides a consistent API for interacting with multiple
Large Language Model providers including OpenAI, Anthropic, Google,
Cohere, Mistral, and Hugging Face.
"""

from .base import LLMProvider, LLMResponse
from .manager import LLMManager
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    CohereProvider,
    MistralProvider,
    HuggingFaceProvider
)

__version__ = "1.0.0"
__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMManager",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "CohereProvider",
    "MistralProvider",
    "HuggingFaceProvider",
]
