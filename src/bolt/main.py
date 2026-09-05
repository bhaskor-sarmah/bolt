"""Main entry point for the Bolt CLI application.

This file defines the Typer CLI application and the `do` command which
accepts a task string and executes it using the autonomous agent.
"""

import typer
from rich.console import Console
from bolt.agent import agent

# Create Typer app with help text
app = typer.Typer(help="Autonomous CLI Assistant", no_args_is_help=False, epilog="An interactive session can be started by running the CLI without any commands.")
# Create Rich console for styled terminal output
console = Console()

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Run the interactive REPL if no command is provided."""
    # If a subcommand like 'do' wasn't invoked, start the REPL
    if ctx.invoked_subcommand is None:
        repl()

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

def repl():
    """Start an interactive REPL session"""
    console.print("[bold green]Welcome to Bolt! Type 'exit' or 'quit' to stop.[/bold green]")
    
    while True:
        try:
            # Display the > prompt and get user input
            task = console.input("[bold blue]>[/bold blue] ").strip()
            
            if not task:
                continue
                
            if task.lower() in ("exit", "quit"):
                break
                
            with console.status("[cyan]Agent is thinking...", spinner="dots"):
                result = agent.run_sync(task)

            console.print(f"[bold green]Result:[/bold green] {result.output}")
            
        except KeyboardInterrupt:
            # Gracefully handle Ctrl+C
            console.print("\n[yellow]Type 'exit' to quit or press Ctrl+D.[/yellow]")
        except EOFError:
            # Gracefully handle Ctrl+D
            console.print()
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    # Run the Typer application when script is executed directly
    app()