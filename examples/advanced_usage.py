"""
Advanced usage examples for LLMs-All-in-One.

This script demonstrates advanced features like model comparison,
fallback mechanisms, and custom configurations.
"""

import os
from dotenv import load_dotenv
from llm_interface import LLMManager, LLMProviderType, Message

# Load environment variables
load_dotenv()


def example_provider_fallback():
    """Example of implementing a fallback mechanism."""
    print("\n" + "="*60)
    print("Provider Fallback Example")
    print("="*60)

    prompt = "What is the Fibonacci sequence?"

    # Define fallback chain
    providers_to_try = [
        ("OpenAI", lambda: LLMManager.create_openai(model="gpt-4o-mini")),
        ("Anthropic", lambda: LLMManager.create_anthropic()),
        ("Google", lambda: LLMManager.create_google()),
    ]

    response = None
    for name, provider_factory in providers_to_try:
        try:
            print(f"\nTrying {name}...")
            provider = provider_factory()
            response = provider.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.7
            )
            print(f"Success! Using {name}")
            print(f"Response: {response.content[:100]}...")
            break
        except Exception as e:
            print(f"{name} failed: {e}")
            continue

    if response is None:
        print("\nAll providers failed!")


def example_temperature_comparison():
    """Compare responses with different temperature settings."""
    print("\n" + "="*60)
    print("Temperature Comparison Example")
    print("="*60)

    provider = LLMManager.create_openai(model="gpt-4o-mini")
    prompt = "Write a creative opening line for a sci-fi novel."

    temperatures = [0.0, 0.5, 1.0]

    for temp in temperatures:
        response = provider.generate(
            prompt=prompt,
            max_tokens=50,
            temperature=temp
        )
        print(f"\nTemperature {temp}:")
        print(f"{response.content}")


def example_batch_processing():
    """Process multiple prompts efficiently."""
    print("\n" + "="*60)
    print("Batch Processing Example")
    print("="*60)

    provider = LLMManager.create_anthropic()

    prompts = [
        "Translate 'Hello, World!' to Spanish",
        "Translate 'Hello, World!' to French",
        "Translate 'Hello, World!' to German",
        "Translate 'Hello, World!' to Japanese",
    ]

    print("\nProcessing multiple translations:")

    for i, prompt in enumerate(prompts, 1):
        response = provider.generate(
            prompt=prompt,
            max_tokens=50,
            temperature=0.3
        )
        print(f"\n{i}. {prompt}")
        print(f"   Answer: {response.content}")


def example_token_usage_tracking():
    """Track token usage across multiple requests."""
    print("\n" + "="*60)
    print("Token Usage Tracking Example")
    print("="*60)

    provider = LLMManager.create_openai(model="gpt-4o-mini")

    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    total_tokens = 0
    results = []

    for prompt in prompts:
        response = provider.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        results.append(response)

        if response.usage:
            total_tokens += response.usage.get("total_tokens", 0)

    print("\nResults:")
    for i, (prompt, response) in enumerate(zip(prompts, results), 1):
        print(f"\n{i}. {prompt}")
        print(f"   Tokens: {response.usage}")

    print(f"\nTotal tokens used: {total_tokens}")


def example_model_listing():
    """List available models for each provider."""
    print("\n" + "="*60)
    print("Available Models Example")
    print("="*60)

    providers = {
        "OpenAI": LLMManager.create_openai(),
        "Anthropic": LLMManager.create_anthropic(),
        "Google": LLMManager.create_google(),
        "Cohere": LLMManager.create_cohere(),
        "Mistral": LLMManager.create_mistral(),
        "HuggingFace": LLMManager.create_huggingface(),
    }

    for name, provider in providers.items():
        try:
            models = provider.get_available_models()
            print(f"\n{name} Models:")
            for model in models:
                print(f"  - {model}")
        except Exception as e:
            print(f"\n{name}: Could not retrieve models - {e}")


def example_custom_configuration():
    """Example with custom provider configurations."""
    print("\n" + "="*60)
    print("Custom Configuration Example")
    print("="*60)

    # OpenAI with custom settings
    provider = LLMManager.create_provider(
        LLMProviderType.OPENAI,
        model="gpt-4o-mini",
        # Custom configurations can be added here
    )

    response = provider.generate(
        prompt="Explain async/await in one sentence.",
        max_tokens=100,
        temperature=0.5,
        top_p=0.9,  # Custom parameter
    )

    print(f"\nResponse: {response.content}")
    print(f"Model: {response.model}")


def main():
    """Run all advanced examples."""
    print("LLMs-All-in-One: Advanced Usage Examples")
    print("=========================================")

    examples = [
        ("Fallback", example_provider_fallback),
        ("Temperature", example_temperature_comparison),
        ("Batch", example_batch_processing),
        ("Token Tracking", example_token_usage_tracking),
        ("Model Listing", example_model_listing),
        ("Custom Config", example_custom_configuration),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n{name} example failed: {e}")


if __name__ == "__main__":
    main()
