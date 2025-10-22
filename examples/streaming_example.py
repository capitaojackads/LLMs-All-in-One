"""
Streaming examples for LLMs-All-in-One.

This script demonstrates how to stream responses from different
LLM providers for real-time output.
"""

import os
import sys
from dotenv import load_dotenv
from llm_interface import LLMManager

# Load environment variables
load_dotenv()


def example_basic_streaming():
    """Example of basic streaming output."""
    print("\n" + "="*60)
    print("Basic Streaming Example")
    print("="*60)

    # Create provider
    provider = LLMManager.create_openai(model="gpt-4o-mini")

    prompt = "Write a short story about a robot learning to paint."

    print("\nStreaming response:")
    print("-" * 60)

    # Stream the response
    for chunk in provider.stream(
        prompt=prompt,
        max_tokens=300,
        temperature=0.8
    ):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 60)


def example_streaming_with_provider_comparison():
    """Compare streaming from different providers."""
    print("\n" + "="*60)
    print("Streaming Comparison: Different Providers")
    print("="*60)

    prompt = "Explain the concept of recursion in programming."

    providers = [
        ("OpenAI", LLMManager.create_openai(model="gpt-4o-mini")),
        ("Anthropic", LLMManager.create_anthropic(model="claude-3-5-haiku-20241022")),
    ]

    for name, provider in providers:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            for chunk in provider.stream(
                prompt=prompt,
                max_tokens=200,
                temperature=0.7
            ):
                print(chunk, end="", flush=True)
            print("\n" + "-" * 60)
        except Exception as e:
            print(f"Error: {e}")


def example_streaming_with_typing_effect():
    """Stream with a typing effect."""
    print("\n" + "="*60)
    print("Streaming with Typing Effect")
    print("="*60)

    import time

    provider = LLMManager.create_anthropic()

    prompt = "Write a motivational quote about perseverance."

    print("\nResponse:")
    print("-" * 60)

    for chunk in provider.stream(
        prompt=prompt,
        max_tokens=100,
        temperature=0.9
    ):
        print(chunk, end="", flush=True)
        time.sleep(0.02)  # Add slight delay for typing effect

    print("\n" + "-" * 60)


def example_streaming_to_file():
    """Stream response directly to a file."""
    print("\n" + "="*60)
    print("Streaming to File Example")
    print("="*60)

    provider = LLMManager.create_google()

    prompt = "Write a comprehensive guide on Python list comprehensions with examples."

    output_file = "streaming_output.txt"

    print(f"\nStreaming response to {output_file}...")

    with open(output_file, "w") as f:
        for chunk in provider.stream(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        ):
            f.write(chunk)
            print(".", end="", flush=True)

    print(f"\n\nOutput saved to {output_file}")

    # Read and display the file
    with open(output_file, "r") as f:
        content = f.read()
        print("\nGenerated content:")
        print("-" * 60)
        print(content)
        print("-" * 60)


def main():
    """Run all streaming examples."""
    print("LLMs-All-in-One: Streaming Examples")
    print("====================================")

    try:
        example_basic_streaming()
    except Exception as e:
        print(f"Basic streaming example failed: {e}")

    try:
        example_streaming_with_provider_comparison()
    except Exception as e:
        print(f"Provider comparison example failed: {e}")

    try:
        example_streaming_with_typing_effect()
    except Exception as e:
        print(f"Typing effect example failed: {e}")

    try:
        example_streaming_to_file()
    except Exception as e:
        print(f"File streaming example failed: {e}")


if __name__ == "__main__":
    main()
