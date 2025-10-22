"""
Google Gemini provider implementation.
"""

import os
from typing import List, Generator, Optional
from ..base import LLMProvider, LLMResponse, Message


class GoogleProvider(LLMProvider):
    """Google Gemini provider implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro", **kwargs):
        """
        Initialize Google provider.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Model to use (default: gemini-1.5-pro)
            **kwargs: Additional configuration
        """
        super().__init__(api_key or os.getenv("GOOGLE_API_KEY"), **kwargs)
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Google client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError(
                    "Google Generative AI package not installed. "
                    "Install with: pip install google-generativeai"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Google Gemini."""
        genai = self._get_client()

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config
        )

        response = model.generate_content(prompt)

        return LLMResponse(
            content=response.text,
            provider=self.provider_name,
            model=self.model,
            usage={
                "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else None,
                "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else None,
                "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None,
            },
            metadata={"finish_reason": response.candidates[0].finish_reason if response.candidates else None}
        )

    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion using Google Gemini."""
        genai = self._get_client()

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config
        )

        # Convert messages to Gemini format
        chat = model.start_chat(history=[])

        # Process all messages except the last one as history
        for i, msg in enumerate(messages[:-1]):
            if msg.role == "user":
                chat.send_message(msg.content)

        # Send the last message and get response
        response = chat.send_message(messages[-1].content)

        return LLMResponse(
            content=response.text,
            provider=self.provider_name,
            model=self.model,
            usage={
                "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else None,
                "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else None,
                "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None,
            }
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream completion using Google Gemini."""
        genai = self._get_client()

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config
        )

        response = model.generate_content(prompt, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def get_available_models(self) -> List[str]:
        """Get available Google models."""
        return [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "Google"
