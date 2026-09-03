import typer
from rich.console import Console
from bolt.agent import agent

app = typer.Typer(help="Autonomous CLI Assistant")
console = Console()

@app.command()
def do(task: str = typer.Argument(..., help="The instruction for the agent to execute")):
    """Execute a task autonomously."""
    console.print(f"[bold blue]Goal:[/bold blue] {task}")
    
    try:
        # Show a spinner in the terminal while the agent runs
        with console.status("[cyan]Agent is thinking...", spinner="dots"):
            # Execute the ReAct loop
            result = agent.run_sync(task)
            
        # Print the final answer
        console.print(f"[bold green]Result:[/bold green] {result.output}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    app()