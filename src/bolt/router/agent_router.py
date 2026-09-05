class AgentRouter:
    """Routes tasks to different models and manages the active model state."""
    
    def __init__(self, console_instance, agent_instance):
        self.console = console_instance
        self.agent = agent_instance
        
        # Configure your available models here
        self.models = {
            "default": "Your standard agent model",
            "gpt-4o": "OpenAI GPT-4o high reasoning",
            "claude-3-sonnet": "Anthropic Claude 3.5 Sonnet",
            "gemini-1.5-pro": "Google Gemini 1.5 Pro"
        }
        self.current_model = "default"

    def list_models(self):
        """Displays available models in the terminal."""
        self.console.print("\n[bold]Available Models:[/bold]")
        for name, description in self.models.items():
            # Highlight the currently active model with an asterisk
            if name == self.current_model:
                self.console.print(f"  [bold green]* {name}[/bold green] - {description} (active)")
            else:
                self.console.print(f"    [cyan]{name}[/cyan] - {description}")
        self.console.print()

    def switch_model(self, model_name: str):
        """Switches the active model if it exists."""
        if model_name in self.models:
            self.current_model = model_name
            self.console.print(f"[bold green]✓ Switched active model to:[/bold green] {model_name}")
        else:
            self.console.print(f"[bold red]Error:[/bold red] Model '{model_name}' not found. Use /models to see options.")

    def run_task(self, task: str):
        """Executes the task using the currently selected model."""
        # Note: You will need to modify your agent.run_sync() method in bolt.agent
        # to accept a 'model' parameter, or initialize a new agent here.
        return self.agent.run_sync(task, model=self.current_model)