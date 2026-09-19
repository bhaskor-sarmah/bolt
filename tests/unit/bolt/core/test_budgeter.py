import pytest
from bolt.core.budgeter import TokenBudgeter
from bolt.core.schemas import UserMessage, AssistantMessage, ToolCall

def test_token_budgeter_initialization():
    # Test valid percentages
    budgeter = TokenBudgeter(
        model_name="gpt-4o",
        max_context_window=1000,
        system_pct=0.10,
        scratchpad_pct=0.60,
        compacted_pct=0.20,
        headroom_pct=0.10
    )

    assert budgeter.system_budget == 100
    assert budgeter.scratchpad_budget == 600
    assert budgeter.compacted_budget == 200
    assert budgeter.headroom_budget == 100

def test_token_budgeter_invalid_percentages():
    with pytest.raises(ValueError, match="Budget percentages must sum to 1.0"):
        TokenBudgeter(system_pct=0.5, scratchpad_pct=0.5, compacted_pct=0.5, headroom_pct=0.5)

def test_count_tokens():
    budgeter = TokenBudgeter()

    # "Hello, world!" is exactly 4 tokens in cl100k_base
    count = budgeter.count_tokens("Hello, world!")
    assert count == 4

    assert budgeter.count_tokens("") == 0
    assert budgeter.count_tokens(None) == 0

def test_count_message_tokens():
    budgeter = TokenBudgeter()

    msg = UserMessage(content="Hello")
    count = budgeter.count_message_tokens(msg)

    # 3 base tokens + 'user' (1 token) + 'Hello' (1 token)
    assert count == 5

    # Test with tool calls
    tool_call_msg = AssistantMessage(
        content="Here is a tool",
        tool_calls=[
            ToolCall(id="call_123", name="my_tool", arguments={"arg1": "val1"})
        ]
    )

    tc_count = budgeter.count_message_tokens(tool_call_msg)
    assert tc_count > 5 # Should be higher due to tool call overhead

def test_check_budget_no_compaction_needed():
    budgeter = TokenBudgeter(max_context_window=1000, scratchpad_pct=0.5)
    # Scratchpad budget is 500 tokens.
    # Threshold is 80% of 500 = 400 tokens.

    messages = [UserMessage(content="Hello world!")]
    status = budgeter.check_budget(messages)

    assert status["needs_compaction"] is False
    assert status["total_tokens"] < 400

def test_check_budget_compaction_needed():
    budgeter = TokenBudgeter(max_context_window=100, scratchpad_pct=0.5)
    # Scratchpad budget is 50.
    # Threshold is 80% of 50 = 40 tokens.

    # Create a message with ~50 tokens
    long_string = "word " * 50
    messages = [UserMessage(content=long_string)]

    status = budgeter.check_budget(messages)

    assert status["needs_compaction"] is True
    assert status["total_tokens"] > 40
