"""
Human in the loop (HITL) security engine (Tier 0/1/2 risk rules)
"""

from enum import Enum
from typing import Dict, Any
from rich.console import Console
from rich.prompt import Confirm

console = Console()

class RiskTier(Enum):
    READ_ONLY = 0    # Auto-approved (e.g., getting the time)
    MUTATING = 1     # Requires standard confirmation
    DESTRUCTIVE = 2  # Requires stern confirmation

class PolicyEngine:
    """Evaluates tool execution requests against security constraints."""
    
    def __init__(self):
        # We hardcode the policy for Phase 2. 
        # In the future, this could be loaded from a YAML configuration.
        self._policies: Dict[str, RiskTier] = {
            "run_shell_command": RiskTier.DESTRUCTIVE
        }

    def _get_tier(self, tool_name: str) -> RiskTier:
        # Default to MUTATING if unknown, to fail secure.
        return self._policies.get(tool_name, RiskTier.MUTATING)

    def request_approval(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Acts as the HITL security gate. Blocks until the human decides."""
        tier = self._get_tier(tool_name)
        
        if tier == RiskTier.READ_ONLY:
            return True
            
        console.print(f"\n[bold yellow]⚠️ Intercepted Tool Request: {tool_name}[/bold yellow]")
        console.print(f"[dim]Payload: {arguments}[/dim]")
        
        # Explicit messaging based on the risk tier
        if tier == RiskTier.DESTRUCTIVE:
            console.print("[bold red]DANGER: This action is destructive and un-sandboxed.[/bold red]")
        elif tier == RiskTier.MUTATING:
            console.print("[bold magenta]Notice: This action will modify state or configurations.[/bold magenta]")
            
        # Blocks the terminal until the user types 'y' or 'n'
        return Confirm.ask("Allow execution?", default=False)