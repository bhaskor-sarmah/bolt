"""Model configuration for the Bolt CLI application.

This file defines the available models and their configurations.
Models can be configured here or loaded from external sources like
environment variables, configuration files, or databases.
"""

from typing import Dict, Any


# Available models configuration
# Format: {model_key: {model_string: str, description: str, **kwargs}}
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "default": {
        "model_string": "openai:qwen3.5:9b",
        "description": "Your standard agent model (Qwen 3.5 9B via Ollama)"
    },
    "gpt-4o": {
        "model_string": "openai:gpt-4o",
        "description": "OpenAI GPT-4o high reasoning model"
        # Note: For OpenAI models, you would need to set OPENAI_API_KEY
        # to a real key (not the dummy one used for Ollama)
    },
    "claude-3-sonnet": {
        "model_string": "anthropic:claude-3-5-sonnet-20241022",
        "description": "Anthropic Claude 3.5 Sonnet"
        # Note: For Anthropic models, you would need to set ANTHROPIC_API_KEY
    },
    "gemini-1.5-pro": {
        "model_string": "google-gla:gemini-1.5-pro",
        "description": "Google Gemini 1.5 Pro"
        # Note: For Google models, you would need to set GOOGLE_API_KEY
    }
}


def get_model_details(model_key: str) -> Dict[str, Any]:
    """Get configuration for a specific model.

    Args:
        model_key: The key identifier for the model (e.g., 'default', 'gpt-4o')

    Returns:
        Dictionary containing model configuration

    Raises:
        KeyError: If the model_key is not found in AVAILABLE_MODELS
    """
    if model_key not in AVAILABLE_MODELS:
        raise KeyError(f"Model '{model_key}' not found. Available models: {list(AVAILABLE_MODELS.keys())}")

    return AVAILABLE_MODELS[model_key]


def list_available_models() -> Dict[str, str]:
    """Get a list of all available models with their descriptions.

    Returns:
        Dictionary mapping model keys to their descriptions
    """
    return {
        key: config["description"] for key, config in AVAILABLE_MODELS.items()
    }


def is_model_available(model_key: str) -> bool:
    """Check if a model is available in the configuration.

    Args:
        model_key: The key identifier for the model

    Returns:
        True if the model is available, False otherwise
    """
    return model_key in AVAILABLE_MODELS