"""
app.py

Main entry point for the Bolt CLI.
"""
import asyncio
import os
import typer
from rich.console import Console

from bolt.adapters.providers.openai import OpenAIAdapter
from bolt.core.schemas import SystemMessage, UserMessage

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
def test_ollama(
    prompt: str = typer.Argument(..., help="The prompt to send to the local LLM"),
    model: str = typer.Option("qwen3.5:latest", help="The local Ollama model to use")
):
    """Test the CLI using a local Ollama instance (Free & Offline)."""
    
    # Ollama's local OpenAI-compatible endpoint
    local_base_url = "http://localhost:11434/v1"
    
    # The OpenAI Python SDK requires an API key string, but Ollama ignores it.
    dummy_api_key = "ollama-local"

    async def run_chat():
        # Notice we reuse the exact same adapter!
        driver = OpenAIAdapter(model_name=model, api_key=dummy_api_key, base_url=local_base_url)
        
        messages = [
            SystemMessage(content="You are a senior systems engineer. Keep responses concise and technical."),
            UserMessage(content=prompt)
        ]

        console.print(f"[dim]Streaming response locally from {model}...[/dim]\n")
        
        async for chunk in driver.stream_generate(messages):
            if chunk.text_delta:
                console.print(chunk.text_delta, end="")
        
        console.print("\n\n[bold green]✓ Local streaming complete.[/bold green]")

    asyncio.run(run_chat())

if __name__ == "__main__":
    app()