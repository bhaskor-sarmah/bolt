"""
ReAct state machine & execution invariants
"""

import json
import logging
from enum import Enum
from typing import Dict, List
from rich.console import Console

from bolt.core.schemas import AssistantMessage, Message, FinishReason, ToolCall, ToolResultMessage
from bolt.core.dispatcher import ResilientDispatcher
from bolt.core.toolregistry import ToolRegistry
from bolt.core.policy import PolicyEngine

console = Console()
logger = logging.getLogger(__name__)

class AgentState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    AWAITING_TOOL = "awaiting_tool"
    EXECUTING = "executing"
    TERMINATED = "terminated"

class ReActAgentFSM:
    """Deterministic loop for managing agent lifecycles and tool invocations."""
    
    def __init__(
        self,
        dispatcher: ResilientDispatcher,
        registry: ToolRegistry,
        policy: PolicyEngine,
        max_steps: int = 10
    ):
        self.dispatcher = dispatcher
        self.registry = registry
        self.policy = policy
        self.max_steps = max_steps
        self.state = AgentState.IDLE

    async def run(self, conversation: List[Message]) -> List[Message]:
        """Executes the autonomous loop until completion or max_steps is hit."""
        self.state = AgentState.PLANNING
        steps = 0
        
        while steps < self.max_steps:
            steps += 1
            
            # --- 1. THE STREAMING ACCUMULATOR ---
            in_thoughts = False
            printed_prefix = False
            full_text = ""
            full_reasoning = ""
            raw_tools: Dict[int, Dict[str, str]] = {} # index -> {id, name, args}
            
            console.print() # Clean newline before starting
            
            async for chunk in self.dispatcher.stream_generate(
                conversation, 
                tools=self.registry.get_definitions()
            ):
                # UI: Handle Reasoning
                if chunk.reasoning_delta:
                    # Print without markup parsing, applying the style directly to the chunk
                    console.print(chunk.reasoning_delta, style="dim italic", end="", markup=False)
                    full_reasoning += chunk.reasoning_delta
                    in_thoughts = True
                    
                # UI: Handle Text
                elif chunk.text_delta:
                    if in_thoughts:
                        console.print("\n") # Add a newline when transitioning from thoughts to text
                        in_thoughts = False
                        
                    if not printed_prefix:
                        console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
                        printed_prefix = True
                        
                    # Print without markup parsing so LLM code snippets don't break Rich
                    console.print(chunk.text_delta, end="", markup=False)
                    full_text += chunk.text_delta
                    
                # BACKGROUND: Accumulate Tool Calls
                if chunk.tool_call_delta:
                    tc = chunk.tool_call_delta
                    idx = tc["index"]
                    
                    if idx not in raw_tools:
                        raw_tools[idx] = {"id": tc.get("id") or "", "name": tc.get("name") or "", "args": ""}
                        
                    if tc.get("arguments"):
                        raw_tools[idx]["args"] += tc["arguments"]

            # Cleanup UI formatting if stream ended while still in thoughts
            if in_thoughts:
                console.print() # Just a newline, no markup tags!
            
            console.print()

            # --- 2. ASSEMBLE THE DOMAIN OBJECT ---
            domain_tool_calls = []
            for idx, raw in raw_tools.items():
                try:
                    args_dict = json.loads(raw["args"]) if raw["args"] else {}
                except json.JSONDecodeError:
                    args_dict = {} # Fallback if model hallucinated invalid JSON, the tool will be called with empty args and likely fail with proper error handling.
                    
                domain_tool_calls.append(ToolCall(
                    id=raw["id"],
                    name=raw["name"],
                    arguments=args_dict
                ))
                
            assistant_msg = AssistantMessage(
                content=full_text if full_text else None,
                reasoning=full_reasoning if full_reasoning else None,
                tool_calls=domain_tool_calls
            )
            conversation.append(assistant_msg)
            
            # --- 3. EVALUATE & TRANSITION ---
            if not domain_tool_calls:
                # If no tools were called, the model finished naturally.
                self.state = AgentState.TERMINATED
                break
                
            self.state = AgentState.AWAITING_TOOL
            
            for tool_call in domain_tool_calls:
                # HITL Gate
                is_approved = self.policy.request_approval(tool_call.name, tool_call.arguments)
                
                self.state = AgentState.EXECUTING
                if is_approved:
                    console.print("[dim]Executing...[/dim]")
                    result = await self.registry.execute(tool_call.name, tool_call.arguments)
                    console.print(f"[dim]Result size: {len(result)} bytes[/dim]")
                else:
                    result = "SYSTEM OVERRIDE: User denied execution."
                    console.print("[dim red]Execution denied.[/dim red]")
                    
                conversation.append(ToolResultMessage(
                    tool_call_id=tool_call.id,
                    content=result
                ))
                
            self.state = AgentState.PLANNING
                
        if steps >= self.max_steps:
            console.print("[bold red]FSM Terminated: Reached max execution steps.[/bold red]")
            self.state = AgentState.TERMINATED
            
        return conversation