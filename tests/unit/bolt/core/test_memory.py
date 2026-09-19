import pytest
from unittest.mock import AsyncMock

from bolt.core.memory import MemoryManager
from bolt.core.budgeter import TokenBudgeter
from bolt.core.schemas import UserMessage, AssistantMessage, SystemMessage, ModelResponse, UsageMetrics, FinishReason
from bolt.ports.driver import ModelDriver


class MockDriverForMemory(ModelDriver):
    def __init__(self):
        super().__init__("mock", "mock")
        self.generate_mock = AsyncMock()

    async def generate(self, messages, tools=None, temperature=0.7, max_tokens=1024):
        # We simulate the driver returning a summary
        response = ModelResponse(
            message=AssistantMessage(content="This is a summary of the past context."),
            usage=UsageMetrics(),
            finish_reason=FinishReason.STOP
        )
        await self.generate_mock(messages, tools, temperature, max_tokens)
        return response

    async def stream_generate(self, messages, tools=None, temperature=0.7, max_tokens=1024):
        pass

    async def close(self):
        pass

@pytest.fixture
def mock_driver():
    return MockDriverForMemory()

@pytest.fixture
def memory_manager(mock_driver):
    budgeter = TokenBudgeter(max_context_window=1000) # Small budget for testing
    return MemoryManager(budgeter=budgeter, driver=mock_driver)


def test_memory_add_message(memory_manager):
    assert len(memory_manager.scratchpad) == 0
    memory_manager.add_message(UserMessage(content="Hello"))
    assert len(memory_manager.scratchpad) == 1

def test_get_full_context(memory_manager):
    memory_manager.set_system_prompt("System Rule")
    memory_manager.compacted_history = AssistantMessage(content="Past Summary")
    memory_manager.add_message(UserMessage(content="New Message"))

    context = memory_manager.get_full_context()

    assert len(context) == 3
    assert isinstance(context[0], SystemMessage)
    assert context[1].content == "Past Summary"
    assert context[2].content == "New Message"


@pytest.mark.asyncio
async def test_compact(memory_manager, mock_driver):
    # Add messages to scratchpad
    for i in range(5):
         memory_manager.add_message(UserMessage(content=f"Message {i}"))

    assert len(memory_manager.scratchpad) == 5

    await memory_manager.compact()

    # Assert driver was called
    mock_driver.generate_mock.assert_called_once()

    # Assert compacted history is updated
    assert memory_manager.compacted_history is not None
    assert "This is a summary of the past context." in memory_manager.compacted_history.content

    # Assert scratchpad was truncated (keeps the last 2)
    assert len(memory_manager.scratchpad) == 2
    assert memory_manager.scratchpad[0].content == "Message 3"
    assert memory_manager.scratchpad[1].content == "Message 4"


@pytest.mark.asyncio
async def test_check_and_compact_breach(memory_manager, mock_driver):
    # Make budgeter very small so it always breaches
    memory_manager.budgeter = TokenBudgeter(max_context_window=10)

    # Add messages
    for i in range(5):
         memory_manager.add_message(UserMessage(content=f"Message {i}"))

    await memory_manager.check_and_compact()

    # Assert compaction triggered
    mock_driver.generate_mock.assert_called_once()


@pytest.mark.asyncio
async def test_check_and_compact_no_breach(memory_manager, mock_driver):
    # Make budgeter very large so it never breaches
    memory_manager.budgeter = TokenBudgeter(max_context_window=10000)

    # Add just 1 small message
    memory_manager.add_message(UserMessage(content=f"Small message"))

    await memory_manager.check_and_compact()

    # Assert compaction NOT triggered
    mock_driver.generate_mock.assert_not_called()
