"""Agent configuration and tool definitions for the Bolt autonomous assistant.

This file sets up the Pydantic-AI agent that connects to a local Ollama
instance via the OpenAI-compatible API, and defines the `get_system_info`
tool that the agent can use to answer questions about the host system.
"""

import os
import platform
import datetime
from pydantic_ai import Agent

# Configure environment variables for Ollama via OpenAI-compatible API
# OPENAI_BASE_URL points to Ollama's OpenAPI-compatible endpoint
if "OPENAI_BASE_URL" not in os.environ:
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
# OPENAI_API_KEY is required by the OpenAI client but Ollama doesn't use it;
# we provide a dummy value to satisfy the client's requirements
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-ollama"

# Initialize the Pydantic-AI agent with the Ollama model via OpenAI provider
agent = Agent(
    'openai:qwen3.5:9b',  # Model name: qwen3.5:9b served by Ollama
    system_prompt="You are a helpful, concise CLI assistant. Use tools when necessary."
)


@agent.tool_plain
def get_system_info() -> str:
    """Returns the current operating system and time. Use this when asked about system info or time."""
    # Get the operating system name (e.g., Darwin, Linux, Windows)
    os_name = platform.system()
    # Get current timestamp formatted as YYYY-MM-DD HH:MM:SS
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"OS: {os_name}, Current Time: {current_time}"