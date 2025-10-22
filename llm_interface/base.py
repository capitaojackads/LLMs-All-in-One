"""
Base abstract classes for LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Generator
from enum import Enum


class LLMProviderType(Enum):
    """Enum for supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMResponse:
    """Standardized response object from LLM providers."""
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.content


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # "user", "assistant", "system"
    content: str


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    All provider implementations must inherit from this class and
    implement the required methods.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the LLM provider.

        Args:
            api_key: API key for the provider (optional, can use env vars)
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse object containing the generated text
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a chat completion.

        Args:
            messages: List of Message objects
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse object containing the generated text
        """
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream a completion for the given prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of generated text
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.

        Returns:
            List of model names
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider."""
        pass

    def validate_api_key(self) -> bool:
        """
        Validate that the API key is set and valid.

        Returns:
            True if API key is valid, False otherwise
        """
        return self.api_key is not None and len(self.api_key) > 0
