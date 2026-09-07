from abc import ABC, abstractmethod
from typing import Any, Dict

from bolt.core.schemas import ToolDefinition

class Tool(ABC):
    """Abstract port for all tools the LLM can invoke."""
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Returns the JSON Schema blueprint sent to the LLM."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Runs the tool logic and returns a stringified result for the LLM."""
        pass