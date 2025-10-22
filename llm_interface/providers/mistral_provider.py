"""
Mistral AI provider implementation.
"""

import os
from typing import List, Generator, Optional
from ..base import LLMProvider, LLMResponse, Message


class MistralProvider(LLMProvider):
    """Mistral AI provider implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-large-latest", **kwargs):
        """
        Initialize Mistral provider.

        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            model: Model to use (default: mistral-large-latest)
            **kwargs: Additional configuration
        """
        super().__init__(api_key or os.getenv("MISTRAL_API_KEY"), **kwargs)
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Mistral client."""
        if self._client is None:
            try:
                from mistralai import Mistral
                self._client = Mistral(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Mistral package not installed. Install with: pip install mistralai"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Mistral."""
        client = self._get_client()

        response = client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            provider=self.provider_name,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            metadata={"finish_reason": response.choices[0].finish_reason}
        )

    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion using Mistral."""
        client = self._get_client()

        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = client.chat.complete(
            model=self.model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            provider=self.provider_name,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            metadata={"finish_reason": response.choices[0].finish_reason}
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream completion using Mistral."""
        client = self._get_client()

        stream = client.chat.stream(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        for chunk in stream:
            if chunk.data.choices[0].delta.content:
                yield chunk.data.choices[0].delta.content

    def get_available_models(self) -> List[str]:
        """Get available Mistral models."""
        return [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b",
        ]

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "Mistral"
