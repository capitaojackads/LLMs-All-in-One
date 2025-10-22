"""
LLMs All-in-One - Streamlit Interface

A unified web interface for interacting with multiple LLM providers.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from llm_interface import LLMManager, LLMProviderType, Message

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LLMs All-in-One",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #0E1117;
    }

    /* Chat message styling */
    .user-message {
        background-color: #1E3A5F;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4A90E2;
    }

    .assistant-message {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #50C878;
    }

    .system-message {
        background-color: #2D2D2D;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.9em;
        color: #B0B0B0;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }

    /* Metrics styling */
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "provider" not in st.session_state:
        st.session_state.provider = None

    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0


def get_provider_models(provider_name):
    """Get available models for a provider."""
    models = {
        "OpenAI": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "Anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
                      "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "Google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        "Cohere": ["command-r-plus", "command-r", "command", "command-light"],
        "Mistral": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
                    "open-mistral-7b", "open-mixtral-8x7b"],
        "HuggingFace": ["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct",
                        "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
    }
    return models.get(provider_name, [])


def create_provider(provider_name, api_key, model):
    """Create an LLM provider instance."""
    try:
        provider_map = {
            "OpenAI": LLMManager.create_openai,
            "Anthropic": LLMManager.create_anthropic,
            "Google": LLMManager.create_google,
            "Cohere": LLMManager.create_cohere,
            "Mistral": LLMManager.create_mistral,
            "HuggingFace": LLMManager.create_huggingface,
        }

        create_func = provider_map.get(provider_name)
        if not create_func:
            return None

        # Create provider with API key if provided
        if api_key:
            return create_func(api_key=api_key, model=model)
        else:
            return create_func(model=model)
    except Exception as e:
        st.error(f"Erro ao criar provedor: {str(e)}")
        return None


def display_chat_message(role, content, metadata=None):
    """Display a chat message with styling."""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 Você:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Assistente:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

        if metadata:
            with st.expander("📊 Detalhes da Resposta"):
                col1, col2, col3 = st.columns(3)
                if "provider" in metadata:
                    col1.metric("Provedor", metadata["provider"])
                if "model" in metadata:
                    col2.metric("Modelo", metadata["model"])
                if "usage" in metadata and metadata["usage"]:
                    usage = metadata["usage"]
                    if "total_tokens" in usage:
                        col3.metric("Tokens", usage["total_tokens"])
    elif role == "system":
        st.markdown(f"""
        <div class="system-message">
            <strong>⚙️ Sistema:</strong> {content}
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 LLMs All-in-One</h1>
        <p style="color: white; font-size: 1.2em;">Interface Unificada para Múltiplos Provedores de LLM</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configurações")

        # Provider selection
        st.subheader("🔌 Provedor")
        provider_name = st.selectbox(
            "Selecione o provedor de LLM:",
            ["OpenAI", "Anthropic", "Google", "Cohere", "Mistral", "HuggingFace"],
            help="Escolha o provedor de inteligência artificial que deseja usar"
        )

        # Model selection
        models = get_provider_models(provider_name)
        selected_model = st.selectbox(
            "Modelo:",
            models,
            help="Escolha o modelo específico do provedor"
        )

        # API Key input
        st.subheader("🔑 API Key")
        env_var_name = f"{provider_name.upper()}_API_KEY"
        if provider_name == "HuggingFace":
            env_var_name = "HUGGINGFACE_API_KEY"

        default_key = os.getenv(env_var_name, "")
        api_key_display = "****" + default_key[-4:] if default_key and len(default_key) > 4 else ""

        api_key = st.text_input(
            "API Key:",
            type="password",
            value=default_key,
            help=f"Sua API key do {provider_name}. Se não fornecida, usará a variável de ambiente {env_var_name}",
            placeholder=api_key_display or "Cole sua API key aqui"
        )

        # Generation parameters
        st.subheader("🎛️ Parâmetros de Geração")

        temperature = st.slider(
            "Temperatura:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controla a aleatoriedade. Valores mais baixos = mais determinístico, valores mais altos = mais criativo"
        )

        max_tokens = st.slider(
            "Máximo de Tokens:",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Número máximo de tokens a serem gerados na resposta"
        )

        use_streaming = st.checkbox(
            "Streaming",
            value=True,
            help="Mostrar a resposta em tempo real conforme ela é gerada"
        )

        # Chat mode
        st.subheader("💬 Modo de Chat")
        chat_mode = st.checkbox(
            "Modo Conversação",
            value=True,
            help="Manter histórico de mensagens para conversações em múltiplas rodadas"
        )

        if not chat_mode:
            st.session_state.messages = []

        # System prompt
        use_system_prompt = st.checkbox(
            "Usar Prompt do Sistema",
            value=False,
            help="Adicionar instruções de sistema para guiar o comportamento do assistente"
        )

        system_prompt = ""
        if use_system_prompt:
            system_prompt = st.text_area(
                "Prompt do Sistema:",
                value="Você é um assistente útil, preciso e educado.",
                help="Instruções que definem o comportamento do assistente"
            )

        # Clear chat button
        st.divider()
        if st.button("🗑️ Limpar Histórico", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_count = 0
            st.session_state.total_tokens = 0
            st.rerun()

        # Statistics
        st.divider()
        st.subheader("📊 Estatísticas")
        st.metric("Mensagens", len(st.session_state.messages))
        st.metric("Conversas", st.session_state.conversation_count)
        st.metric("Tokens Totais", st.session_state.total_tokens)

    # Main chat interface
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"💬 Conversação com {provider_name}")

    with col2:
        st.info(f"🎯 Modelo: **{selected_model}**")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            display_chat_message(
                msg.get("role"),
                msg.get("content"),
                msg.get("metadata")
            )

    # User input
    st.divider()

    # Create columns for input and button
    input_col, button_col = st.columns([5, 1])

    with input_col:
        user_input = st.text_area(
            "Sua mensagem:",
            placeholder="Digite sua mensagem aqui...",
            key="user_input",
            height=100,
            label_visibility="collapsed"
        )

    with button_col:
        st.write("")  # Spacing
        st.write("")  # Spacing
        send_button = st.button("📤 Enviar", use_container_width=True, type="primary")

    # Process user input
    if send_button and user_input:
        # Validate API key
        if not api_key:
            st.error(f"❌ Por favor, forneça uma API key para {provider_name}")
            return

        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Create provider
        with st.spinner(f"🔄 Criando conexão com {provider_name}..."):
            provider = create_provider(provider_name, api_key, selected_model)

        if not provider:
            st.error("❌ Falha ao criar o provedor. Verifique suas credenciais.")
            return

        # Generate response
        try:
            if chat_mode:
                # Build messages for chat
                messages = []

                # Add system prompt if enabled
                if use_system_prompt and system_prompt:
                    messages.append(Message(role="system", content=system_prompt))

                # Add conversation history
                for msg in st.session_state.messages:
                    if msg["role"] in ["user", "assistant"]:
                        messages.append(Message(
                            role=msg["role"],
                            content=msg["content"]
                        ))

                if use_streaming:
                    # Streaming response
                    with st.spinner("🤖 Gerando resposta..."):
                        response_placeholder = st.empty()
                        full_response = ""

                        # Note: Streaming in chat mode requires special handling
                        # For simplicity, we'll use generate with the last message
                        for chunk in provider.stream(
                            prompt=user_input,
                            max_tokens=max_tokens,
                            temperature=temperature
                        ):
                            full_response += chunk
                            response_placeholder.markdown(f"""
                            <div class="assistant-message">
                                <strong>🤖 Assistente:</strong><br>
                                {full_response}
                            </div>
                            """, unsafe_allow_html=True)

                        # Add to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "metadata": {
                                "provider": provider_name,
                                "model": selected_model
                            }
                        })
                else:
                    # Non-streaming response
                    with st.spinner("🤖 Gerando resposta..."):
                        response = provider.chat(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )

                        # Add to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.content,
                            "metadata": {
                                "provider": response.provider,
                                "model": response.model,
                                "usage": response.usage
                            }
                        })

                        # Update statistics
                        if response.usage and "total_tokens" in response.usage:
                            st.session_state.total_tokens += response.usage["total_tokens"]

            else:
                # Simple generation mode
                if use_streaming:
                    response_placeholder = st.empty()
                    full_response = ""

                    for chunk in provider.stream(
                        prompt=user_input,
                        max_tokens=max_tokens,
                        temperature=temperature
                    ):
                        full_response += chunk
                        response_placeholder.markdown(f"""
                        <div class="assistant-message">
                            <strong>🤖 Assistente:</strong><br>
                            {full_response}
                        </div>
                        """, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "metadata": {
                            "provider": provider_name,
                            "model": selected_model
                        }
                    })
                else:
                    with st.spinner("🤖 Gerando resposta..."):
                        response = provider.generate(
                            prompt=user_input,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.content,
                            "metadata": {
                                "provider": response.provider,
                                "model": response.model,
                                "usage": response.usage
                            }
                        })

                        if response.usage and "total_tokens" in response.usage:
                            st.session_state.total_tokens += response.usage["total_tokens"]

            # Increment conversation count
            st.session_state.conversation_count += 1

            # Success message
            st.success("✅ Resposta gerada com sucesso!")

            # Rerun to update the chat
            st.rerun()

        except Exception as e:
            st.error(f"❌ Erro ao gerar resposta: {str(e)}")
            st.info("💡 Dica: Verifique se sua API key está correta e se você tem créditos disponíveis.")

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤖 <strong>LLMs All-in-One</strong> - Interface Unificada para Múltiplos Provedores de LLM</p>
        <p>Suporta: OpenAI • Anthropic • Google • Cohere • Mistral • HuggingFace</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
