from typing import Dict, List, Any
from bolt.ports.tool import Tool
from bolt.core.schemas import ToolDefinition

class ToolRegistry:
    """Centralized manager and router for all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        
    def register(self, tool: Tool) -> None:
        """Registers a tool using the name from its definition."""
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = tool
        
    def get_definitions(self) -> List[ToolDefinition]:
        """Returns the list of tool blueprints to send to the LLM."""
        return [tool.definition for tool in self._tools.values()]
        
    async def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """Routes an execution request to the appropriate tool adapter."""
        if name not in self._tools:
            # We return a string instead of raising an exception because the 
            # LLM needs to see this error to know it hallucinated a tool name.
            return f"Error: Tool '{name}' does not exist or is not registered."
            
        tool = self._tools[name]
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            # Catch execution errors so they don't crash the entire CLI. 
            # Feed the error back to the LLM so it can attempt a fix.
            return f"Execution error in '{name}': {str(e)}"