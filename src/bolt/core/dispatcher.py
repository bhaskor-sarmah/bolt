"""
dispatcher.py

A Decorator that wraps any ModelDriver with Circuit Breaker logic.
This separates resilience (a domain concern) from the adapter code.
"""

from typing import List, Optional, AsyncGenerator
from bolt.core.schemas import Message, ToolDefinition, ModelResponse, StreamChunk
from bolt.ports.driver import ModelDriver
from bolt.core.resilience import CircuitBreaker

class ResilientDispatcher(ModelDriver):
    def __init__(self, driver: ModelDriver, circuit_breaker: Optional[CircuitBreaker] = None):
        # Pass the underlying driver's credentials up to satisfy the abstract base class
        super().__init__(driver.model_name, driver.api_key, driver.base_url)
        
        self.driver = driver
        self.circuit_breaker = circuit_breaker or CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Routes standard generation through the Circuit Breaker."""
        return await self.circuit_breaker.call(
            self.driver.generate, 
            messages, tools, temperature, max_tokens
        )

    async def stream_generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Routes streaming generation through the Circuit Breaker."""
        async for chunk in self.circuit_breaker.stream_call(
            self.driver.stream_generate, 
            messages, tools, temperature, max_tokens
        ):
            yield chunk

    async def close(self) -> None:
        """Delegates graceful shutdown to the underlying driver."""
        await self.driver.close()