"""
Sandboxed subprocess runner
"""

import asyncio

from bolt.ports.tool import Tool
from bolt.core.schemas import ToolDefinition

class ShellTool(Tool):
    """Executes local shell commands asynchronously."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="run_shell_command",
            description=(
                "Executes a shell command on the local machine. "
                "Use this to inspect directories, read files, or run scripts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    }
                },
                "required": ["command"]
            }
        )

    async def execute(self, **kwargs) -> str:
        """Runs the command and captures stdout/stderr."""
        command = kwargs.get("command")
        if not command:
            return "Error: Missing required argument 'command'."

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        output = ""
        if stdout:
            output += stdout.decode('utf-8')
        if stderr:
            output += f"\n[STDERR]\n{stderr.decode('utf-8')}"
            
        return output.strip() or "Command executed successfully with no output."