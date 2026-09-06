"""
app.py

Main entry point for the Bolt CLI.
"""
import asyncio
import os
import typer
from dotenv import load_dotenv
from rich.console import Console

from bolt.adapters.providers.openai import OpenAIAdapter
from bolt.core.schemas import SystemMessage, UserMessage

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
        driver = OpenAIAdapter(model_name=model, api_key=api_key, base_url=local_base_url)
        try:
            messages = [
                SystemMessage(content="You are a senior systems engineer. Keep responses concise and technical."),
                UserMessage(content=prompt)
            ]

            console.print(f"[dim]Streaming response via LiteLLM ({model})...[/dim]\n")
            
            async for chunk in driver.stream_generate(messages):
                if chunk.text_delta:
                    console.print(chunk.text_delta, end="")
            
            console.print("\n\n[bold green]✓ LiteLLM streaming complete.[/bold green]")
            
        finally:
            await driver.close()

    asyncio.run(run_chat())

if __name__ == "__main__":
    app()