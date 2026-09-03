import os
import platform
import datetime
from pydantic_ai import Agent

# Explicitly tell Pydantic-AI where to find your local Ollama instance (via OpenAI-compatible API)
if "OPENAI_BASE_URL" not in os.environ:
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-ollama"

# Initialize the agent
agent = Agent(
    'openai:qwen3.5:9b',
    system_prompt="You are a helpful, concise CLI assistant. Use tools when necessary."
)

@agent.tool_plain
def get_system_info() -> str:
    """Returns the current operating system and time. Use this when asked about system info or time."""
    os_name = platform.system()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"OS: {os_name}, Current Time: {current_time}"