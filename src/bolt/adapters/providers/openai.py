"""
openai.py

Concrete adapter for the OpenAI API.
Implements the ModelDriver port to map our domain schemas to OpenAI's expected JSON format.
"""

import json
from typing import List, Optional, AsyncGenerator, Dict, Any
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from bolt.core.schemas import (
    Message, Role, ModelResponse, StreamChunk, 
    AssistantMessage, ToolCall, UsageMetrics, FinishReason, ToolDefinition
)
from bolt.ports.driver import ModelDriver


class OpenAIAdapter(ModelDriver):
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        super().__init__(model_name, api_key, base_url)
        # We use the Async client to ensure we don't block the terminal UI
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Translates our domain Message objects into OpenAI's dictionary format."""
        formatted = []
        for msg in messages:
            # Base payload for all messages
            payload: Dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content or ""
            }

            # Handle Assistant tool calls
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            # OpenAI requires arguments to be a JSON string, not a dict
                            "arguments": json.dumps(tc.arguments) 
                        }
                    }
                    for tc in msg.tool_calls
                ]

            # Handle Tool Results returning to the model
            elif msg.role == Role.TOOL:
                payload["tool_call_id"] = getattr(msg, "tool_call_id", "")

            formatted.append(payload)
            
        return formatted

    def _format_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Translates our ToolDefinitions into OpenAI's tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in tools
        ]

    def _map_finish_reason(self, reason: Optional[str]) -> FinishReason:
        """Maps OpenAI's finish reasons to our domain Enum."""
        if not reason:
            return FinishReason.STOP
        
        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "content_filter": FinishReason.ERROR,
        }
        return mapping.get(reason, FinishReason.STOP)

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """Executes a standard async completion."""
        
        kwargs = {
            "model": self.model_name,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        # Execute the network call
        raw_response: ChatCompletion = await self.client.chat.completions.create(**kwargs)
        choice = raw_response.choices[0]
        openai_msg = choice.message

        # Translate OpenAI's tool calls back to our domain schema
        domain_tool_calls = []
        if openai_msg.tool_calls:
            for tc in openai_msg.tool_calls:
                if tc.type == "function":
                    domain_tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            # Parse their JSON string back into a Python dictionary
                            arguments=json.loads(tc.function.arguments)
                        )
                    )

        # Build our domain AssistantMessage
        assistant_msg = AssistantMessage(
            content=openai_msg.content,
            tool_calls=domain_tool_calls
        )

        # Extract usage metrics
        usage = UsageMetrics()
        if raw_response.usage:
            usage = UsageMetrics(
                prompt_tokens=raw_response.usage.prompt_tokens,
                completion_tokens=raw_response.usage.completion_tokens,
                total_tokens=raw_response.usage.total_tokens
            )

        # Return our unified domain object
        return ModelResponse(
            message=assistant_msg,
            usage=usage,
            finish_reason=self._map_finish_reason(choice.finish_reason)
        )

    async def stream_generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Executes a streaming completion for the terminal UI."""
        
        kwargs = {
            "model": self.model_name,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        stream = await self.client.chat.completions.create(**kwargs)

        async with stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta
                
                text_content = getattr(delta, "content", None)
                
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)

                # OpenAI streaming tool calls are complex (they stream JSON fragments).
                # For Phase 1, we focus on streaming text. Tool chunking comes later.
                if text_content or reasoning:
                    yield StreamChunk(
                        text_delta=text_content,
                        reasoning_delta=reasoning
                    )


    async def close(self) -> None:
        """Closes the underlying AsyncOpenAI HTTP connection pool."""
        await self.client.close()