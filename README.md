# LLMs All-in-One

A unified Python interface for all major Large Language Model (LLM) providers. Seamlessly switch between OpenAI, Anthropic, Google, Cohere, Mistral, and Hugging Face with a single, consistent API.

## Features

- **🌐 Web Interface**: Beautiful Streamlit app for easy interaction with all LLM providers
- **Unified API**: One consistent interface for all LLM providers
- **Easy Provider Switching**: Change providers with a single line of code
- **Consistent Response Format**: Standardized output across all providers
- **Streaming Support**: Real-time response streaming from all providers
- **Chat Conversations**: Multi-turn conversations with context management
- **Type Safety**: Full type hints for better IDE support
- **Extensible**: Easy to add custom providers
- **Well Documented**: Comprehensive examples and documentation

## Supported Providers

| Provider | Models | Status |
|----------|--------|--------|
| **OpenAI** | GPT-4, GPT-4 Turbo, GPT-3.5 | ✅ Supported |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku | ✅ Supported |
| **Google** | Gemini 1.5 Pro, Gemini 1.5 Flash | ✅ Supported |
| **Cohere** | Command R+, Command R, Command | ✅ Supported |
| **Mistral** | Mistral Large, Medium, Small | ✅ Supported |
| **Hugging Face** | Llama 3, Mixtral, Gemma, Falcon | ✅ Supported |

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### For development

```bash
pip install -e ".[dev]"
```

## 🌐 Web Interface (Streamlit)

The easiest way to get started is with our beautiful web interface!

### Launch the Web App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Web Interface Features

- 🎨 **Modern Dark Theme**: Beautiful, eye-friendly interface
- 🔌 **Provider Selection**: Easy dropdown to switch between all 6 LLM providers
- 🎯 **Model Selection**: Choose from available models for each provider
- 🔑 **API Key Management**: Secure input with environment variable support
- 💬 **Chat History**: Full conversation history with context
- ⚡ **Real-time Streaming**: See responses as they're generated
- 🎛️ **Adjustable Parameters**: Control temperature, max tokens, and more
- 📊 **Usage Statistics**: Track messages, conversations, and tokens
- 🌍 **Portuguese Interface**: Fully localized UI

![Streamlit Interface](https://via.placeholder.com/800x400?text=LLMs+All-in-One+Interface)

## Quick Start (Python API)

### 1. Set up your API keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
# ... etc
```

### 2. Basic Usage

```python
from llm_interface import LLMManager

# Create a provider (OpenAI example)
provider = LLMManager.create_openai(model="gpt-4")

# Generate a response
response = provider.generate(
    prompt="Explain quantum computing in simple terms.",
    max_tokens=200,
    temperature=0.7
)

print(response.content)
```

### 3. Switch Providers Easily

```python
from llm_interface import LLMManager

# Use OpenAI
openai_provider = LLMManager.create_openai(model="gpt-4")

# Use Anthropic
anthropic_provider = LLMManager.create_anthropic(model="claude-3-5-sonnet-20241022")

# Use Google
google_provider = LLMManager.create_google(model="gemini-1.5-pro")

# All have the same interface!
for provider in [openai_provider, anthropic_provider, google_provider]:
    response = provider.generate("Hello, who are you?")
    print(f"{provider.provider_name}: {response.content}")
```

## Examples

### Simple Text Generation

```python
from llm_interface import LLMManager

provider = LLMManager.create_openai()

response = provider.generate(
    prompt="Write a haiku about coding",
    max_tokens=100,
    temperature=0.9
)

print(response.content)
```

### Chat Conversations

```python
from llm_interface import LLMManager, Message

provider = LLMManager.create_anthropic()

messages = [
    Message(role="system", content="You are a helpful coding assistant."),
    Message(role="user", content="How do I reverse a list in Python?"),
]

response = provider.chat(messages=messages, max_tokens=200)
print(response.content)
```

### Streaming Responses

```python
from llm_interface import LLMManager

provider = LLMManager.create_google()

for chunk in provider.stream(
    prompt="Tell me a story about AI",
    max_tokens=500
):
    print(chunk, end="", flush=True)
```

### Provider Fallback

```python
from llm_interface import LLMManager

def generate_with_fallback(prompt):
    """Try multiple providers until one succeeds."""
    providers = [
        LLMManager.create_openai,
        LLMManager.create_anthropic,
        LLMManager.create_google,
    ]

    for create_provider in providers:
        try:
            provider = create_provider()
            return provider.generate(prompt)
        except Exception as e:
            print(f"Provider failed: {e}")
            continue

    raise Exception("All providers failed")

response = generate_with_fallback("What is machine learning?")
print(response.content)
```

### Comparing Models

```python
from llm_interface import LLMManager

prompt = "Explain recursion in one sentence."

providers = {
    "GPT-4": LLMManager.create_openai(model="gpt-4"),
    "Claude": LLMManager.create_anthropic(model="claude-3-5-sonnet-20241022"),
    "Gemini": LLMManager.create_google(model="gemini-1.5-pro"),
}

for name, provider in providers.items():
    response = provider.generate(prompt, max_tokens=100)
    print(f"\n{name}:")
    print(response.content)
```

## Advanced Usage

### Custom Configuration

```python
from llm_interface import LLMManager, LLMProviderType

# Create provider with custom settings
provider = LLMManager.create_provider(
    LLMProviderType.OPENAI,
    api_key="your-api-key",
    model="gpt-4",
)

response = provider.generate(
    prompt="Hello!",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,  # Provider-specific parameter
)
```

### Token Usage Tracking

```python
from llm_interface import LLMManager

provider = LLMManager.create_openai()

response = provider.generate("What is Python?", max_tokens=100)

print(f"Tokens used: {response.usage}")
# Output: {'prompt_tokens': 4, 'completion_tokens': 50, 'total_tokens': 54}
```

### List Available Models

```python
from llm_interface import LLMManager

provider = LLMManager.create_openai()
models = provider.get_available_models()

print("Available OpenAI models:")
for model in models:
    print(f"  - {model}")
```

## API Reference

### LLMManager

Factory class for creating LLM provider instances.

#### Methods

- `create_provider(provider_type, api_key=None, model=None, **kwargs)` - Create a provider instance
- `create_openai(api_key=None, model="gpt-4", **kwargs)` - Create OpenAI provider
- `create_anthropic(api_key=None, model="claude-3-5-sonnet-20241022", **kwargs)` - Create Anthropic provider
- `create_google(api_key=None, model="gemini-1.5-pro", **kwargs)` - Create Google provider
- `create_cohere(api_key=None, model="command-r-plus", **kwargs)` - Create Cohere provider
- `create_mistral(api_key=None, model="mistral-large-latest", **kwargs)` - Create Mistral provider
- `create_huggingface(api_key=None, model="meta-llama/Meta-Llama-3-8B-Instruct", **kwargs)` - Create Hugging Face provider

### LLMProvider

Base interface for all LLM providers.

#### Methods

- `generate(prompt, max_tokens=1000, temperature=0.7, **kwargs)` - Generate a completion
- `chat(messages, max_tokens=1000, temperature=0.7, **kwargs)` - Chat completion
- `stream(prompt, max_tokens=1000, temperature=0.7, **kwargs)` - Stream a completion
- `get_available_models()` - Get list of available models
- `provider_name` - Get the provider name

#### Parameters

- `prompt` (str): The input prompt
- `messages` (List[Message]): List of chat messages
- `max_tokens` (int): Maximum tokens to generate (default: 1000)
- `temperature` (float): Sampling temperature 0.0-1.0 (default: 0.7)
- `**kwargs`: Additional provider-specific parameters

### LLMResponse

Standardized response object.

#### Attributes

- `content` (str): The generated text
- `provider` (str): Name of the provider
- `model` (str): Model used
- `usage` (dict): Token usage information
- `metadata` (dict): Additional metadata

### Message

Represents a chat message.

#### Attributes

- `role` (str): Message role ("user", "assistant", "system")
- `content` (str): Message content

## Examples Directory

Check out the `examples/` directory for more comprehensive examples:

- `basic_usage.py` - Basic examples with different providers
- `chat_example.py` - Multi-turn conversation examples
- `streaming_example.py` - Streaming response examples
- `advanced_usage.py` - Advanced features and configurations

Run an example:

```bash
cd examples
python basic_usage.py
```

## Project Structure

```
LLMs-All-in-One/
├── llm_interface/           # Main package
│   ├── __init__.py         # Package initialization
│   ├── base.py             # Base classes and interfaces
│   ├── manager.py          # LLM manager/factory
│   └── providers/          # Provider implementations
│       ├── openai_provider.py
│       ├── anthropic_provider.py
│       ├── google_provider.py
│       ├── cohere_provider.py
│       ├── mistral_provider.py
│       └── huggingface_provider.py
├── examples/               # Usage examples
│   ├── basic_usage.py
│   ├── chat_example.py
│   ├── streaming_example.py
│   └── advanced_usage.py
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Requirements

- Python 3.8+
- API keys for the providers you want to use

### Provider SDK Requirements

The package will automatically handle the required SDKs for each provider:

- OpenAI: `openai>=1.0.0`
- Anthropic: `anthropic>=0.34.0`
- Google: `google-generativeai>=0.3.0`
- Cohere: `cohere>=5.0.0`
- Mistral: `mistralai>=1.0.0`
- Hugging Face: `huggingface-hub>=0.20.0`

## Getting API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google**: https://makersuite.google.com/app/apikey
- **Cohere**: https://dashboard.cohere.com/api-keys
- **Mistral**: https://console.mistral.ai/
- **Hugging Face**: https://huggingface.co/settings/tokens

## Contributing

Contributions are welcome! Here are some ways you can contribute:

- Add support for new LLM providers
- Improve documentation and examples
- Report bugs and issues
- Suggest new features

## License

MIT License - see LICENSE file for details

## Acknowledgments

This project provides a unified interface for various LLM providers. Each provider's SDK and API is owned by their respective companies:

- OpenAI (GPT models)
- Anthropic (Claude models)
- Google (Gemini models)
- Cohere (Command models)
- Mistral AI (Mistral models)
- Hugging Face (Various open models)

## Support

For issues, questions, or contributions, please visit the GitHub repository.

## Roadmap

- [ ] Async/await support for all providers
- [ ] Batch processing optimization
- [ ] Cost tracking and estimation
- [ ] Response caching
- [ ] Retry logic with exponential backoff
- [ ] Rate limiting support
- [ ] Embedding support
- [ ] Image generation support
- [ ] Function calling support
- [ ] More provider integrations (Replicate, Together.ai, etc.)

---

Made with ❤️ for the AI community
