"""Agent factory for creating Pydantic-AI agent instances."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent

# Import Pydantic-AI model classes
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel

# Import Providers
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider

from bolt.model.config import get_model_details

# Define where the user's config file will live
CONFIG_DIR = Path.home() / ".bolt"
ENV_FILE = CONFIG_DIR / ".env"

def load_user_config():
    """Load configuration from the user's home directory."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)

def get_pydantic_ai_model(model_string: str):
    """Instantiate the correct Pydantic-AI model based on the string and config."""
    load_user_config()
    
    if model_string.startswith("openai:") or model_string.startswith("ollama:"):
        actual_model = model_string.split(":", 1)[-1] if ":" in model_string else model_string
        
        base_url = os.getenv("OLLAMA_BASE_URL") if "ollama" in model_string else os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OLLAMA_API_KEY", "dummy-key") if "ollama" in model_string else os.getenv("OPENAI_API_KEY")
        
        # Use OpenAIChatModel for standard Chat Completions API
        return OpenAIChatModel(
            model_name=actual_model,
            provider=OpenAIProvider(
                api_key=api_key,
                base_url=base_url
            )
        )

    elif model_string.startswith("anthropic:"):
        actual_model = model_string.split(":", 1)[-1]
        return AnthropicModel(
            model_name=actual_model,
            provider=AnthropicProvider(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        )

    elif model_string.startswith("google-gla:") or model_string.startswith("gemini:") or model_string.startswith("google:"):
        actual_model = model_string.split(":", 1)[-1]
        return GoogleModel(
            model_name=actual_model,
            provider=GoogleProvider(
                api_key=os.getenv("GOOGLE_API_KEY")
            )
        )
        
    else:
        # Fallback to letting Pydantic-AI parse the raw string natively
        return model_string

def create_agent(model_key: str) -> Agent:
    """Create a Pydantic-AI agent instance for the specified model."""
    model_detail = get_model_details(model_key)
    model_string = model_detail["model_string"]

    # Get the explicitly configured model object
    configured_model = get_pydantic_ai_model(model_string)

    return Agent(
        model=configured_model,
        system_prompt="You are a helpful, concise CLI assistant. Use tools when necessary."
    )

def create_agent_with_custom_prompt(model_key: str, system_prompt: str) -> Agent:
    """Create a Pydantic-AI agent instance with a custom system prompt."""
    model_detail = get_model_details(model_key)
    model_string = model_detail["model_string"]

    configured_model = get_pydantic_ai_model(model_string)

    return Agent(
        model=configured_model,
        system_prompt=system_prompt
    )