"""
driver.py

The abstract Port (Interface) for all LLM providers (Hexagonal Architecture).
Purpose: This acts as a strict contract. Any provider we want to add to the CLI 
(e.g., OpenAIAdapter, AnthropicAdapter, OllamaAdapter) MUST implement these exact methods.
Your core application will only ever talk to this interface, never to the adapters directly.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, AsyncIterator
from bolt.core.schemas import Message, ToolDefinition, ModelResponse, StreamChunk

class ModelDriver(ABC):
    """
    ABC (Abstract Base Class) ensures that Python will throw an error if you 
    try to create an adapter but forget to write the generate() or stream_generate() methods.
    """

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        """
        Initializes the connection details for the provider.
        """
        self.model_name = model_name   # e.g., "gpt-4o" or "claude-3-5-sonnet-20240620"
        self.api_key = api_key         # Injected from your ~/.bolt/config.json
        self.base_url = base_url       # Useful for overriding with local proxies (like Ollama or LiteLLM proxy)

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """
        Executes a standard, non-streaming completion.
        
        Purpose: Used for background tasks where the user doesn't need to see the text 
        appearing in real-time (e.g., the Phase 3 Memory Compaction agent summarizing history).
        
        Args:
            messages: The full conversation history up to this point.
            tools: The list of Python functions the model is allowed to use.
            temperature: Creativity/randomness (0.0 is deterministic, 1.0 is creative).
            max_tokens: A safety cap on how much the model is allowed to generate.
            
        Returns:
            A clean ModelResponse object that your FSM can process.
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        """
        Executes a streaming completion.
        
        Purpose: Used for the main user-facing REPL. The AsyncIterator allows your 
        Typer/Rich CLI app to `async for chunk in driver.stream_generate(...)` and print 
        text to the screen instantly as the model types it out.
        """
        yield StreamChunk()

    @abstractmethod
    async def close(self) -> None:
        """
        Gracefully shuts down any underlying network clients, 
        database connections, or connection pools.
        """
        pass