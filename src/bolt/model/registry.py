"""Model registry for managing agent instances.

This file implements a registry pattern for managing model configurations
and agent instances. It provides a centralized way to access agents for
different models, with caching to avoid recreating agents unnecessarily.
"""

from typing import Dict
from pydantic_ai import Agent

# Import our configuration and factory modules
from bolt.model.config import (
    AVAILABLE_MODELS,
    get_model_details,
    list_available_models,
    is_model_available
)
from bolt.model.factory import create_agent


class ModelRegistry:
    """Registry for managing model configurations and agent instances.

    This class acts as a central registry for all available models.
    It caches agent instances to avoid recreating them unnecessarily
    and provides methods for getting agents and switching between models.
    """

    def __init__(self):
        """Initialize the model registry."""
        # Cache for agent instances: {model_key: agent_instance}
        self._agent_cache: Dict[str, Agent] = {}

        # Track the currently active model
        self._current_model: str = "default"

    def get_agent(self, model_key: str) -> Agent:
        """Get an agent instance for the specified model.

        If the agent for this model has already been created, it returns
        the cached instance. Otherwise, it creates a new one and caches it.

        Args:
            model_key: The key identifier for the model

        Returns:
            Pydantic-AI Agent instance for the specified model

        Raises:
            KeyError: If the model_key is not available
        """
        if not is_model_available(model_key):
            raise KeyError(
                f"Model '{model_key}' is not available. "
                f"Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        # Return cached agent if available
        if model_key in self._agent_cache:
            return self._agent_cache[model_key]

        # Create new agent and cache it
        agent_instance = create_agent(model_key)
        self._agent_cache[model_key] = agent_instance
        return agent_instance

    def get_current_agent(self) -> Agent:
        """Get the agent instance for the currently active model.

        Returns:
            Pydantic-AI Agent instance for the current model
        """
        return self.get_agent(self._current_model)

    def switch_model(self, model_key: str) -> None:
        """Switch the currently active model.

        Args:
            model_key: The key identifier for the model to switch to

        Raises:
            KeyError: If the model_key is not available
        """
        if not is_model_available(model_key):
            raise KeyError(
                f"Model '{model_key}' is not available. "
                f"Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        self._current_model = model_key

    def get_current_model(self) -> str:
        """Get the key of the currently active model.

        Returns:
            String key of the currently active model
        """
        return self._current_model

    def list_available_models(self) -> Dict[str, str]:
        """List all available models with their descriptions.

        Returns:
            Dictionary mapping model keys to their descriptions
        """
        return list_available_models()

    def is_model_available(self, model_key: str) -> bool:
        """Check if a model is available.

        Args:
            model_key: The key identifier for the model

        Returns:
            True if the model is available, False otherwise
        """
        return is_model_available(model_key)

    def clear_cache(self) -> None:
        """Clear the agent instance cache.

        This forces recreation of agent instances on the next request.
        Useful when configuration changes or for testing.
        """
        self._agent_cache.clear()

    def get_cache_info(self) -> Dict[str, bool]:
        """Get information about which models have cached agents.

        Returns:
            Dictionary mapping model keys to whether they're cached
        """
        return {
            model_key: model_key in self._agent_cache
            for model_key in AVAILABLE_MODELS.keys()
        }


# Create a global registry instance for easy access
# This follows the pattern used in your existing agent.py and main.py
model_registry = ModelRegistry()