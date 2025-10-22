"""
Cohere provider implementation.
"""

import os
from typing import List, Generator, Optional
from ..base import LLMProvider, LLMResponse, Message


class CohereProvider(LLMProvider):
    """Cohere provider implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "command-r-plus", **kwargs):
        """
        Initialize Cohere provider.

        Args:
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            model: Model to use (default: command-r-plus)
            **kwargs: Additional configuration
        """
        super().__init__(api_key or os.getenv("COHERE_API_KEY"), **kwargs)
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Cohere package not installed. Install with: pip install cohere"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Cohere."""
        client = self._get_client()

        response = client.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return LLMResponse(
            content=response.generations[0].text,
            provider=self.provider_name,
            model=self.model,
            metadata={
                "finish_reason": response.generations[0].finish_reason,
                "likelihood": response.generations[0].likelihood if hasattr(response.generations[0], 'likelihood') else None
            }
        )

    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion using Cohere."""
        client = self._get_client()

        # Convert messages to Cohere chat format
        chat_history = []
        message = messages[-1].content  # Last message is the current query

        for msg in messages[:-1]:
            chat_history.append({
                "role": "USER" if msg.role == "user" else "CHATBOT",
                "message": msg.content
            })

        response = client.chat(
            model=self.model,
            message=message,
            chat_history=chat_history if chat_history else None,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return LLMResponse(
            content=response.text,
            provider=self.provider_name,
            model=self.model,
            metadata={
                "finish_reason": response.finish_reason if hasattr(response, 'finish_reason') else None
            }
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream completion using Cohere."""
        client = self._get_client()

        response = client.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )

        for event in response:
            if event.event_type == "text-generation":
                yield event.text

    def get_available_models(self) -> List[str]:
        """Get available Cohere models."""
        return [
            "command-r-plus",
            "command-r",
            "command",
            "command-light",
        ]

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "Cohere"
