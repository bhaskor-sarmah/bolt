"""
app.py

Main entry point for the Bolt CLI.
"""

import typer
from rich.console import Console

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

if __name__ == "__main__":
    app()