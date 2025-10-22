"""
Basic usage examples for LLMs-All-in-One.

This script demonstrates how to use different LLM providers
with a unified interface.
"""

import os
from dotenv import load_dotenv
from llm_interface import LLMManager, LLMProviderType

# Load environment variables from .env file
load_dotenv()


def example_openai():
    """Example using OpenAI GPT."""
    print("\n" + "="*60)
    print("OpenAI GPT-4 Example")
    print("="*60)

    # Create OpenAI provider
    provider = LLMManager.create_openai(model="gpt-4o-mini")

    # Generate a simple completion
    response = provider.generate(
        prompt="Explain quantum computing in one sentence.",
        max_tokens=100,
        temperature=0.7
    )

    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Response: {response.content}")
    print(f"Usage: {response.usage}")


def example_anthropic():
    """Example using Anthropic Claude."""
    print("\n" + "="*60)
    print("Anthropic Claude Example")
    print("="*60)

    # Create Anthropic provider
    provider = LLMManager.create_anthropic()

    # Generate a completion
    response = provider.generate(
        prompt="What are the three laws of robotics?",
        max_tokens=200,
        temperature=0.7
    )

    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Response: {response.content}")


def example_google():
    """Example using Google Gemini."""
    print("\n" + "="*60)
    print("Google Gemini Example")
    print("="*60)

    # Create Google provider
    provider = LLMManager.create_google(model="gemini-1.5-flash")

    # Generate a completion
    response = provider.generate(
        prompt="Write a haiku about artificial intelligence.",
        max_tokens=100,
        temperature=0.9
    )

    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Response: {response.content}")


def example_multiple_providers():
    """Example comparing responses from multiple providers."""
    print("\n" + "="*60)
    print("Comparing Multiple Providers")
    print("="*60)

    prompt = "What is the meaning of life?"

    # Create multiple providers
    providers = [
        LLMManager.create_openai(model="gpt-4o-mini"),
        LLMManager.create_anthropic(model="claude-3-5-haiku-20241022"),
        LLMManager.create_google(model="gemini-1.5-flash"),
    ]

    # Get responses from all providers
    for provider in providers:
        try:
            response = provider.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )
            print(f"\n{response.provider} ({response.model}):")
            print(f"{response.content}")
        except Exception as e:
            print(f"\n{provider.provider_name}: Error - {str(e)}")


def main():
    """Run all examples."""
    print("LLMs-All-in-One: Basic Usage Examples")
    print("======================================")

    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY not set")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set")
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set")

    # Run examples (comment out providers you don't have API keys for)
    try:
        example_openai()
    except Exception as e:
        print(f"OpenAI example failed: {e}")

    try:
        example_anthropic()
    except Exception as e:
        print(f"Anthropic example failed: {e}")

    try:
        example_google()
    except Exception as e:
        print(f"Google example failed: {e}")

    try:
        example_multiple_providers()
    except Exception as e:
        print(f"Multiple providers example failed: {e}")


if __name__ == "__main__":
    main()
