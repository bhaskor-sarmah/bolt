"""
app.py

Main entry point for the Bolt CLI.
"""
import asyncio
import os
from typing import List
from rich.prompt import Prompt
import typer
from contextlib import aclosing
from dotenv import load_dotenv
from rich.console import Console

from bolt.adapters.providers.openai import OpenAIAdapter
from bolt.adapters.tools.shell import ShellTool
import uuid
from bolt.core.fsm import ReActAgentFSM
from bolt.core.policy import PolicyEngine
from bolt.core.resilience import CircuitBreaker
from bolt.core.schemas import Message, SystemMessage, UserMessage
from bolt.core.dispatcher import ResilientDispatcher
from bolt.core.toolregistry import ToolRegistry
from bolt.core.memory import MemoryManager
from bolt.core.budgeter import TokenBudgeter
from bolt.config.settings import settings
from bolt.adapters.storage.sqlite import SQLiteSessionStorage

# Load environment variables from the .env file
load_dotenv()

# Create the Typer app
app = typer.Typer(help="Bolt AI CLI", no_args_is_help=False)
console = Console()

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Run this when the user just types 'bolt' with no arguments."""
    if ctx.invoked_subcommand is None:
        console.print("[bold green]Success![/bold green] The Bolt CLI is installed and running perfectly! 🚀")

@app.command()
def ping():
    """A quick test command to verify routing."""
    console.print("[bold blue]Pong![/bold blue] Subcommands are working.")

@app.command()
def test_litellm(
    prompt: str = typer.Argument(..., help="The prompt to send to the LLM"),
    # Read the default model from .env, with a fallback
    model: str = typer.Option(
        os.getenv("LITELLM_MODEL", "openrouter/nvidia/nemotron-3.5-lightning:free"), 
        help="The LiteLLM model to use (overrides .env)"
    )
):
    """Test the CLI using a LiteLLM proxy deployment."""
    
    # Pull config from .env (falling back to standard LiteLLM port if missing)
    local_base_url = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    api_key = os.getenv("LITELLM_API_KEY")

    # Guardrail to ensure you don't send requests without an API key
    if not api_key:
        console.print("[bold red]Error:[/bold red] LITELLM_API_KEY not found in .env file.")
        raise typer.Exit(code=1)

    async def run_chat():
        
        raw_driver = OpenAIAdapter(model_name=model, api_key=api_key, base_url=local_base_url)
        
        dispatcher = ResilientDispatcher(driver=raw_driver)
        
        try:
            messages = [
                SystemMessage(content="You are a senior systems engineer. Keep responses concise and technical."),
                UserMessage(content=prompt)
            ]

            console.print(f"[dim]Streaming response via Dispatcher ({model})...[/dim]\n")
            
            in_thoughts = False
            
            async with aclosing(dispatcher.stream_generate(messages, max_tokens=4096)) as stream:
                async for chunk in stream:
                    
                    # 1. If we receive a reasoning token, style it dim and italic
                    if chunk.reasoning_delta:
                        if not in_thoughts:
                            # Start formatting the thought block
                            console.print("[dim italic]", end="")
                            in_thoughts = True
                            
                        console.print(chunk.reasoning_delta, style="dim italic", end="")
                        
                    # 2. If we receive a standard text token, print it normally
                    elif chunk.text_delta:
                        if in_thoughts:
                            # The model finished thinking! Transition the UI.
                            console.print("\n\n[bold cyan]Final Answer:[/bold cyan]\n", end="")
                            in_thoughts = False
                            
                        console.print(chunk.text_delta, end="")

            # It prints a final newline so your zsh terminal resets cleanly.
            console.print()
            
        finally:
            await dispatcher.close()

    asyncio.run(run_chat())

@app.command()
def agent(
    model: str = typer.Option(
        os.getenv("LITELLM_MODEL", "openrouter/nvidia/nemotron-3.5-lightning:free"), 
        "--model", "-m", help="The LiteLLM model to use (overrides .env)"
    ),
    max_steps: int = typer.Option(10, "--steps", "-s", help="Max FSM execution steps per prompt"),
    resume: str = typer.Option(None, "--resume", "-r", help="Resume a session by ID")
):
    """Starts the interactive ReAct agent REPL with tool execution."""
    
    local_base_url = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    api_key = os.getenv("LITELLM_API_KEY")

    if not api_key:
        console.print("[bold red]Error:[/bold red] LITELLM_API_KEY not found in .env file.")
        raise typer.Exit(code=1)

    # 1. Initialize Phase 1: Driver & Dispatcher (Targeting LiteLLM)
    driver = OpenAIAdapter(model_name=model, api_key=api_key, base_url=local_base_url)
    circuit_breaker = CircuitBreaker()
    dispatcher = ResilientDispatcher(driver=driver, circuit_breaker=circuit_breaker)

    # 2. Initialize Phase 3: Storage and Memory
    storage = SQLiteSessionStorage(db_path=settings.db_path)
    session_id = resume if resume else str(uuid.uuid4())
    budgeter = TokenBudgeter(max_context_window=4096)
    memory_manager = MemoryManager(budgeter=budgeter, driver=driver, session_id=session_id, storage=storage)

    # 3. Initialize Phase 2: Tools & Registry
    registry = ToolRegistry()
    registry.register(ShellTool())

    # 4. Initialize Phase 2: Policy Engine & FSM
    policy = PolicyEngine()
    fsm = ReActAgentFSM(
        dispatcher=dispatcher,
        registry=registry,
        policy=policy,
        max_steps=max_steps
    )

    async def _repl():
        await storage.initialize()
        await memory_manager.initialize()

        console.print("\n[bold green]⚡ Bolt Agent Initialized[/bold green]")
        console.print(f"[dim]Session ID: {session_id}[/dim]")
        console.print(f"[dim]Model: {model} | Max Steps: {max_steps} | Endpoint: {local_base_url}[/dim]")
        console.print("[dim]Type 'exit' or 'quit' to end the session.[/dim]\n")
        
        # Give the agent its core identity
        memory_manager.set_system_prompt(
            "You are an expert CLI assistant running on a macOS environment. "
            "You have access to a tool named 'run_shell_command'. "
            "Use it to inspect the system, read files, and accomplish tasks. "
            "If asked to check the OS, find the current directory, or list files, "
            "immediately use the tool."
        )

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            except (KeyboardInterrupt, EOFError):
                break
                
            if user_input.strip().lower() in ["exit", "quit"]:
                break
                
            if not user_input.strip():
                continue
                
            memory_manager.add_message(UserMessage(content=user_input))
            
            # Hand control to the Autonomous FSM
            try:
                # The FSM manages the tools and returns the updated conversation history
                updated_conversation = await fsm.run(memory_manager.get_full_context())

                # We need to extract only the new messages that were added to the scratchpad,
                # since get_full_context prepends the system prompt and compacted history.
                # The scratchpad is essentially everything after the system prompt and compacted history.
                num_preamble_msgs = (1 if memory_manager.system_prompt else 0) + (1 if memory_manager.compacted_history else 0)
                memory_manager.scratchpad = updated_conversation[num_preamble_msgs:]

                # Check and compact memory if budget is exceeded
                await memory_manager.check_and_compact()
            except Exception as e:
                console.print(f"\n[bold red]System Error:[/bold red] {e}")

        console.print(f"\n[bold yellow]Session ended. You can resume later with:[/bold yellow] bolt agent --resume {session_id}")

        # Always attempt to save state on graceful exit.
        # If there are uncompacted messages in the scratchpad, force a final compaction.
        if memory_manager.scratchpad:
            await memory_manager.compact()
        else:
            await memory_manager.save_state()

        await storage.close()
        await dispatcher.close()

    # Bridge sync CLI to async core
    asyncio.run(_repl())

if __name__ == "__main__":
    app()