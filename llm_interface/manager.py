"""
Unified LLM manager for creating and managing different LLM providers.
"""

from typing import Dict, Optional, Type
from .base import LLMProvider, LLMProviderType
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    CohereProvider,
    MistralProvider,
    HuggingFaceProvider
)


class LLMManager:
    """
    Unified manager for creating and managing LLM providers.

    This class provides a factory pattern for instantiating different
    LLM providers with a consistent interface.
    """

    # Registry of available providers
    _providers: Dict[LLMProviderType, Type[LLMProvider]] = {
        LLMProviderType.OPENAI: OpenAIProvider,
        LLMProviderType.ANTHROPIC: AnthropicProvider,
        LLMProviderType.GOOGLE: GoogleProvider,
        LLMProviderType.COHERE: CohereProvider,
        LLMProviderType.MISTRAL: MistralProvider,
        LLMProviderType.HUGGINGFACE: HuggingFaceProvider,
    }

    @classmethod
    def create_provider(
        cls,
        provider_type: LLMProviderType,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_type: The type of provider to create
            api_key: API key for the provider (optional, uses env vars if not provided)
            model: Model to use (optional, uses provider default if not provided)
            **kwargs: Additional provider-specific configuration

        Returns:
            An instance of the requested LLM provider

        Raises:
            ValueError: If the provider type is not supported

        Example:
            >>> manager = LLMManager()
            >>> openai = manager.create_provider(
            ...     LLMProviderType.OPENAI,
            ...     model="gpt-4"
            ... )
            >>> response = openai.generate("Hello, world!")
        """
        if provider_type not in cls._providers:
            raise ValueError(
                f"Unsupported provider type: {provider_type}. "
                f"Supported types: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_type]

        # Build kwargs for provider initialization
        init_kwargs = kwargs.copy()
        if api_key is not None:
            init_kwargs["api_key"] = api_key
        if model is not None:
            init_kwargs["model"] = model

        return provider_class(**init_kwargs)

    @classmethod
    def create_openai(
        cls,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        **kwargs
    ) -> OpenAIProvider:
        """Create an OpenAI provider instance."""
        return cls.create_provider(
            LLMProviderType.OPENAI,
            api_key=api_key,
            model=model,
            **kwargs
        )

    @classmethod
    def create_anthropic(
        cls,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        **kwargs
    ) -> AnthropicProvider:
        """Create an Anthropic provider instance."""
        return cls.create_provider(
            LLMProviderType.ANTHROPIC,
            api_key=api_key,
            model=model,
            **kwargs
        )

    @classmethod
    def create_google(
        cls,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-pro",
        **kwargs
    ) -> GoogleProvider:
        """Create a Google provider instance."""
        return cls.create_provider(
            LLMProviderType.GOOGLE,
            api_key=api_key,
            model=model,
            **kwargs
        )

    @classmethod
    def create_cohere(
        cls,
        api_key: Optional[str] = None,
        model: str = "command-r-plus",
        **kwargs
    ) -> CohereProvider:
        """Create a Cohere provider instance."""
        return cls.create_provider(
            LLMProviderType.COHERE,
            api_key=api_key,
            model=model,
            **kwargs
        )

    @classmethod
    def create_mistral(
        cls,
        api_key: Optional[str] = None,
        model: str = "mistral-large-latest",
        **kwargs
    ) -> MistralProvider:
        """Create a Mistral provider instance."""
        return cls.create_provider(
            LLMProviderType.MISTRAL,
            api_key=api_key,
            model=model,
            **kwargs
        )

    @classmethod
    def create_huggingface(
        cls,
        api_key: Optional[str] = None,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        **kwargs
    ) -> HuggingFaceProvider:
        """Create a Hugging Face provider instance."""
        return cls.create_provider(
            LLMProviderType.HUGGINGFACE,
            api_key=api_key,
            model=model,
            **kwargs
        )

    @classmethod
    def list_providers(cls) -> list:
        """Get list of all available provider types."""
        return list(cls._providers.keys())

    @classmethod
    def register_provider(
        cls,
        provider_type: LLMProviderType,
        provider_class: Type[LLMProvider]
    ):
        """
        Register a custom provider.

        Args:
            provider_type: The provider type identifier
            provider_class: The provider class to register
        """
        cls._providers[provider_type] = provider_class
