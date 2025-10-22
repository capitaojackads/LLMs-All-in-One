"""
Anthropic Claude provider implementation.
"""

import os
from typing import List, Generator, Optional
from ..base import LLMProvider, LLMResponse, Message


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", **kwargs):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-3-5-sonnet-20241022)
            **kwargs: Additional configuration
        """
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"), **kwargs)
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Anthropic Claude."""
        client = self._get_client()

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )

        return LLMResponse(
            content=response.content[0].text,
            provider=self.provider_name,
            model=self.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            metadata={"stop_reason": response.stop_reason}
        )

    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion using Anthropic Claude."""
        client = self._get_client()

        # Separate system messages from other messages
        system_message = None
        formatted_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        request_kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": formatted_messages,
            **kwargs
        }

        if system_message:
            request_kwargs["system"] = system_message

        response = client.messages.create(**request_kwargs)

        return LLMResponse(
            content=response.content[0].text,
            provider=self.provider_name,
            model=self.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            metadata={"stop_reason": response.stop_reason}
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream completion using Anthropic Claude."""
        client = self._get_client()

        with client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield text

    def get_available_models(self) -> List[str]:
        """Get available Anthropic models."""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "Anthropic"
