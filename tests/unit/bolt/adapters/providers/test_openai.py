import pytest
from unittest.mock import AsyncMock, MagicMock

from bolt.core.schemas import SystemMessage, UserMessage, AssistantMessage, ToolCall, ToolDefinition
from bolt.adapters.providers.openai import OpenAIAdapter

def test_format_messages():
    """Validates domain Message objects are correctly parsed into OpenAI dictionary schemas."""
    adapter = OpenAIAdapter(model_name="test", api_key="key")
    messages = [
        SystemMessage(content="You are a system."),
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi", tool_calls=[
            ToolCall(id="call_1", name="get_weather", arguments={"loc": "Assam"})
        ])
    ]
    
    formatted = adapter._format_messages(messages)
    
    assert len(formatted) == 3
    assert formatted[0] == {"role": "system", "content": "You are a system."}
    assert formatted[1] == {"role": "user", "content": "Hello"}
    assert formatted[2]["role"] == "assistant"
    assert formatted[2]["content"] == "Hi"
    assert formatted[2]["tool_calls"][0]["id"] == "call_1"
    assert formatted[2]["tool_calls"][0]["function"]["name"] == "get_weather"

def test_format_tools():
    """Validates ToolDefinitions are correctly packed into the OpenAI JSON schema format."""
    adapter = OpenAIAdapter(model_name="test", api_key="key")
    tools = [
        ToolDefinition(
            name="run_command",
            description="Executes a shell command",
            parameters={"type": "object", "properties": {"cmd": {"type": "string"}}}
        )
    ]
    
    formatted = adapter._format_tools(tools)
    
    assert len(formatted) == 1
    assert formatted[0]["type"] == "function"
    assert formatted[0]["function"]["name"] == "run_command"
    assert "cmd" in formatted[0]["function"]["parameters"]["properties"]

async def test_stream_generate_extracts_reasoning_and_text():
    """Mocks the HTTP stream to verify reasoning vs standard text chunks are routed correctly."""
    adapter = OpenAIAdapter(model_name="test", api_key="key")
    
    # 1. Mock the OpenAI delta objects
    mock_delta_1 = MagicMock()
    mock_delta_1.content = None
    mock_delta_1.reasoning_content = "<think> wait..." 
    mock_delta_1.reasoning = None

    mock_delta_2 = MagicMock()
    mock_delta_2.content = "Final Answer"
    mock_delta_2.reasoning_content = None
    mock_delta_2.reasoning = None

    mock_chunk_1 = MagicMock(); mock_chunk_1.choices = [MagicMock(delta=mock_delta_1)]
    mock_chunk_2 = MagicMock(); mock_chunk_2.choices = [MagicMock(delta=mock_delta_2)]
    
    class MockStream:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc_val, exc_tb): pass
        async def __aiter__(self):
            yield mock_chunk_1
            yield mock_chunk_2

    adapter.client = MagicMock()
    adapter.client.chat.completions.create = AsyncMock(return_value=MockStream())
    
    results = [chunk async for chunk in adapter.stream_generate([UserMessage(content="hi")])]
    
    assert len(results) == 2
    assert results[0].reasoning_delta == "<think> wait..."
    assert results[0].text_delta is None
    
    assert results[1].text_delta == "Final Answer"
    assert results[1].reasoning_delta is None