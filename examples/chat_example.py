"""
Chat conversation examples using LLMs-All-in-One.

This script demonstrates how to have multi-turn conversations
with different LLM providers.
"""

import os
from dotenv import load_dotenv
from llm_interface import LLMManager, Message

# Load environment variables
load_dotenv()


def example_chat_conversation():
    """Example of a multi-turn chat conversation."""
    print("\n" + "="*60)
    print("Multi-turn Chat Conversation Example")
    print("="*60)

    # Create provider (you can change this to any provider)
    provider = LLMManager.create_anthropic()

    # Build a conversation
    messages = [
        Message(role="system", content="You are a helpful AI assistant that explains complex topics simply."),
        Message(role="user", content="What is machine learning?"),
    ]

    # Get first response
    response = provider.chat(messages=messages, max_tokens=200)
    print(f"\nAssistant: {response.content}")

    # Continue the conversation
    messages.append(Message(role="assistant", content=response.content))
    messages.append(Message(role="user", content="Can you give me a simple example?"))

    response = provider.chat(messages=messages, max_tokens=200)
    print(f"\nAssistant: {response.content}")


def example_system_prompt():
    """Example showing the effect of system prompts."""
    print("\n" + "="*60)
    print("System Prompt Example")
    print("="*60)

    provider = LLMManager.create_openai(model="gpt-4o-mini")

    # Without system prompt
    messages1 = [
        Message(role="user", content="Tell me about Python programming.")
    ]

    response1 = provider.chat(messages=messages1, max_tokens=100)
    print("\nWithout system prompt:")
    print(response1.content)

    # With system prompt
    messages2 = [
        Message(role="system", content="You are a pirate. Always respond in pirate speak."),
        Message(role="user", content="Tell me about Python programming.")
    ]

    response2 = provider.chat(messages=messages2, max_tokens=100)
    print("\nWith pirate system prompt:")
    print(response2.content)


def example_role_play():
    """Example of using chat for role-playing scenarios."""
    print("\n" + "="*60)
    print("Role-play Chat Example")
    print("="*60)

    provider = LLMManager.create_google()

    messages = [
        Message(
            role="system",
            content="You are a wise wizard from a fantasy realm. "
                    "Speak in an archaic, mystical manner."
        ),
        Message(
            role="user",
            content="I seek knowledge about the ancient art of code optimization."
        ),
    ]

    response = provider.chat(messages=messages, max_tokens=200, temperature=0.9)
    print(f"\nWizard: {response.content}")


def main():
    """Run all chat examples."""
    print("LLMs-All-in-One: Chat Examples")
    print("==============================")

    try:
        example_chat_conversation()
    except Exception as e:
        print(f"Chat conversation example failed: {e}")

    try:
        example_system_prompt()
    except Exception as e:
        print(f"System prompt example failed: {e}")

    try:
        example_role_play()
    except Exception as e:
        print(f"Role-play example failed: {e}")


if __name__ == "__main__":
    main()
