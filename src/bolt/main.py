"""Main entry point for the Bolt CLI application.

This file defines the Typer CLI application and the `do` command which
accepts a task string and executes it using the autonomous agent.
"""

import typer
from rich.console import Console
from bolt.agent import agent

# Create Typer app with help text
app = typer.Typer(help="Autonomous CLI Assistant")
# Create Rich console for styled terminal output
console = Console()


@app.command()
def do(task: str = typer.Argument(..., help="The instruction for the agent to execute")):
    """Execute a task autonomously."""
    # Display the goal/task to the user
    console.print(f"[bold blue]User says:[/bold blue] {task}")

    try:
        # Show a spinner in the terminal while the agent runs
        with console.status("[cyan]Agent is thinking...", spinner="dots"):
            # Execute the ReAct (Reasoning and Acting) loop
            result = agent.run_sync(task)

        # Print the final answer from the agent
        console.print(f"[bold green]Result:[/bold green] {result.output}")

    except Exception as e:
        # Display any errors that occur during agent execution
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@app.command()
def msg(greeting: str = "Hi"):
    """Testing to say hi"""
    console.print(f"[bold green] User saying {greeting}")

if __name__ == "__main__":
    # Run the Typer application when script is executed directly
    app()