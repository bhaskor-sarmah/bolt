import pytest
from typing import AsyncGenerator
from unittest.mock import AsyncMock

from bolt.core.schemas import StreamChunk, ModelResponse, AssistantMessage, UsageMetrics, FinishReason
from bolt.ports.driver import ModelDriver
from bolt.core.dispatcher import ResilientDispatcher

class MockDriver(ModelDriver):
    """A dummy driver to verify the Dispatcher delegates calls correctly."""
    def __init__(self):
        super().__init__("mock-model", "mock-key", "http://mock-url")
        
        # FIX: Instantiate a strictly valid ModelResponse
        valid_response = ModelResponse(
            message=AssistantMessage(content="Mock response"),
            usage=UsageMetrics(),
            finish_reason=FinishReason.STOP
        )
        self.generate_mock = AsyncMock(return_value=valid_response)
        self.close_mock = AsyncMock()

    async def generate(self, messages, tools=None, temperature=0.7, max_tokens=1024):
        return await self.generate_mock(messages, tools, temperature, max_tokens)

    async def stream_generate(self, messages, tools=None, temperature=0.7, max_tokens=1024) -> AsyncGenerator[StreamChunk, None]:
        yield StreamChunk(text_delta="Mock ")
        yield StreamChunk(text_delta="stream")

    async def close(self):
        await self.close_mock()

async def test_dispatcher_delegates_generate(fast_circuit_breaker):
    """Ensures standard generate calls pass through the circuit breaker to the driver."""
    driver = MockDriver()
    dispatcher = ResilientDispatcher(driver=driver, circuit_breaker=fast_circuit_breaker)
    
    response = await dispatcher.generate([])
    
    assert response.message.content == "Mock response"
    driver.generate_mock.assert_called_once()

async def test_dispatcher_delegates_stream_generate(fast_circuit_breaker):
    """Ensures stream_generate yields chunks from the underlying driver."""
    driver = MockDriver()
    dispatcher = ResilientDispatcher(driver=driver, circuit_breaker=fast_circuit_breaker)
    
    chunks = [chunk async for chunk in dispatcher.stream_generate([])]
    
    assert len(chunks) == 2
    assert chunks[0].text_delta == "Mock "
    assert chunks[1].text_delta == "stream"

async def test_dispatcher_delegates_close():
    """Ensures the close lifecycle method is propagated downward."""
    driver = MockDriver()
    dispatcher = ResilientDispatcher(driver=driver)
    
    await dispatcher.close()
    
    driver.close_mock.assert_called_once()