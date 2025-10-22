"""
Hugging Face provider implementation.
"""

import os
from typing import List, Generator, Optional
from ..base import LLMProvider, LLMResponse, Message


class HuggingFaceProvider(LLMProvider):
    """Hugging Face Inference API provider implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/Meta-Llama-3-8B-Instruct", **kwargs):
        """
        Initialize Hugging Face provider.

        Args:
            api_key: Hugging Face API key (defaults to HUGGINGFACE_API_KEY env var)
            model: Model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)
            **kwargs: Additional configuration
        """
        super().__init__(api_key or os.getenv("HUGGINGFACE_API_KEY"), **kwargs)
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Hugging Face client."""
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(token=self.api_key)
            except ImportError:
                raise ImportError(
                    "Hugging Face Hub package not installed. "
                    "Install with: pip install huggingface-hub"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Hugging Face."""
        client = self._get_client()

        response = client.text_generation(
            prompt=prompt,
            model=self.model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            return_full_text=False,
            **kwargs
        )

        return LLMResponse(
            content=response,
            provider=self.provider_name,
            model=self.model,
        )

    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion using Hugging Face."""
        client = self._get_client()

        # Convert messages to chat format
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = client.chat_completion(
            messages=formatted_messages,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            provider=self.provider_name,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None,
                "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
            }
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream completion using Hugging Face."""
        client = self._get_client()

        stream = client.text_generation(
            prompt=prompt,
            model=self.model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            yield chunk

    def get_available_models(self) -> List[str]:
        """Get some popular available Hugging Face models."""
        return [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "google/gemma-7b-it",
            "tiiuae/falcon-180B-chat",
        ]

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "HuggingFace"
