import pytest
from bolt.core.schemas import (
    Role,
    FinishReason,
    ToolCall,
    ToolDefinition,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    UsageMetrics,
    ModelResponse,
    StreamChunk,
)

def test_role_and_finish_reason_enums():
    """Validates enum string mappings."""
    assert Role.SYSTEM == "system"
    assert Role.USER == "user"
    assert Role.ASSISTANT == "assistant"
    assert Role.TOOL == "tool"

    assert FinishReason.STOP == "stop"
    assert FinishReason.LENGTH == "length"
    assert FinishReason.TOOL_CALLS == "tool_calls"
    assert FinishReason.ERROR == "error"

def test_messages_enforce_correct_roles():
    """Ensures specialized message subclasses enforce their respective roles."""
    sys_msg = SystemMessage(content="Be concise.")
    assert sys_msg.role == Role.SYSTEM
    assert sys_msg.content == "Be concise."

    user_msg = UserMessage(content="Check system")
    assert user_msg.role == Role.USER

    asst_msg = AssistantMessage(content="Checking...", tool_calls=[
        ToolCall(id="call_1", name="run_cmd", arguments={"cmd": "uptime"})
    ])
    assert asst_msg.role == Role.ASSISTANT
    assert asst_msg.tool_calls is not None
    assert len(asst_msg.tool_calls) == 1
    assert asst_msg.tool_calls[0].name == "run_cmd"

    tool_msg = ToolResultMessage(content="load average: 0.1", tool_call_id="call_1")
    assert tool_msg.role == Role.TOOL
    assert tool_msg.tool_call_id == "call_1"

def test_tool_definitions():
    """Validates tool schema blueprint formatting."""
    tool_def = ToolDefinition(
        name="run_cmd",
        description="Executes a local command",
        parameters={"type": "object", "properties": {"cmd": {"type": "string"}}}
    )
    assert tool_def.name == "run_cmd"
    assert "cmd" in tool_def.parameters["properties"]

def test_model_response_nested_structure():
    """Validates the aggregated ModelResponse schema."""
    response = ModelResponse(
        message=AssistantMessage(content="All systems operational."),
        usage=UsageMetrics(prompt_tokens=42, completion_tokens=12, total_tokens=54),
        finish_reason=FinishReason.STOP
    )
    
    assert response.message.content == "All systems operational."
    assert response.usage.prompt_tokens == 42
    assert response.usage.total_tokens == 54
    assert response.finish_reason == FinishReason.STOP

def test_stream_chunk_optional_deltas():
    """Validates StreamChunk fields default correctly to None."""
    chunk_text = StreamChunk(text_delta="hello")
    assert chunk_text.text_delta == "hello"
    assert chunk_text.reasoning_delta is None
    assert chunk_text.tool_call_delta is None

    chunk_reasoning = StreamChunk(reasoning_delta="thinking...")
    assert chunk_reasoning.reasoning_delta == "thinking..."
    assert chunk_reasoning.text_delta is None