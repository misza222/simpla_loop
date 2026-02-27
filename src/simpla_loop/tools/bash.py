"""Bash tool for executing shell commands.

WARNING: Executing arbitrary shell commands is dangerous.
This tool should be used with caution and proper sandboxing.

The BashTool provides a controlled way for agents to execute shell
commands. It includes safety features like:
- Timeout to prevent hanging
- Output capture (stdout/stderr)
- Return code checking
- Working directory control

Security Considerations:
- Commands run with Python process permissions
- No input sanitization (command passed directly to shell)
- Can execute any shell command (rm, curl, etc.)
- Network access depends on system configuration

Recommended for:
- Local development and testing
- Sandboxed environments (containers, VMs)
- Trusted code execution scenarios

Not recommended for:
- Production multi-tenant systems
- Exposing to untrusted users
- Environments with sensitive data

Example:
    >>> tool = BashTool(timeout=30)
    >>> result = tool.execute(command="echo 'hello world'")
    >>> result.success
    True
    >>> result.data.stdout
    'hello world\\n'
    >>> result.data.returncode
    0
"""

import subprocess
from dataclasses import dataclass
from typing import Any

from simpla_loop.core.tool import Tool, ToolParameter, ToolResult


@dataclass
class BashResult:
    """Structured result from bash execution.

    BashResult provides a clean interface for accessing command output,
    making it easier to work with than raw subprocess results.

    Attributes:
        returncode: Exit code from the command. 0 typically means success,
                   non-zero indicates an error (depending on the command).
        stdout: Standard output from the command as a string.
               Empty string if no output.
        stderr: Standard error from the command as a string.
               Empty string if no errors.

    Properties:
        success: True if returncode == 0. Convenience property.

    Example:
        >>> result = BashResult(returncode=0, stdout="hello\\n", stderr="")
        >>> result.success
        True
        >>> print(result.stdout)
        hello
    """

    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """True if command exited with code 0.

        Note: Some commands may return non-zero for non-error conditions.
        Check command documentation for specific exit codes.

        Returns:
            Boolean indicating success.

        Example:
            >>> BashResult(0, "", "").success
            True
            >>> BashResult(1, "", "error").success
            False
        """
        return self.returncode == 0


class BashTool(Tool):
    """Execute bash shell commands.

    This tool runs commands in a subprocess and captures output.
    Commands run with a timeout to prevent hanging.

    WARNING: Only use in trusted environments. Commands are
    executed with the permissions of the Python process.

    Configuration:
    - timeout: Maximum seconds to wait for command
    - cwd: Working directory for command execution

    The tool validates that the 'command' parameter is provided
    before execution.

    Example:
        >>> # Basic usage
        >>> tool = BashTool(timeout=30)
        >>> result = tool.execute(command="echo 'hello'")
        >>> print(result.data.stdout)
        "hello\\n"

        >>> # With custom working directory
        >>> tool = BashTool(cwd="/tmp")
        >>> result = tool.execute(command="pwd")
        >>> print(result.data.stdout)
        "/tmp\\n"

        >>> # Error handling
        >>> result = tool.execute(command="exit 1")
        >>> result.success
        False
        >>> result.data.returncode
        1
    """

    def __init__(self, timeout: int = 60, cwd: str | None = None) -> None:
        """Initialize bash tool.

        Args:
            timeout: Maximum seconds to wait for command execution.
                    If exceeded, subprocess.TimeoutExpired is raised
                    and converted to a ToolResult.fail().
                    Default is 60 seconds.
            cwd: Working directory for commands.
                 None means current working directory.
                 Path is passed directly to subprocess.run().

        Returns:
            None

        Example:
            >>> tool = BashTool(timeout=10)
            >>> tool = BashTool(cwd="/home/user/projects")
            >>> tool = BashTool(timeout=5, cwd="/tmp")
        """
        self.timeout = timeout
        self.cwd = cwd

    @property
    def name(self) -> str:
        """Tool identifier.

        Returns:
            'bash' - the canonical name for this tool.
        """
        return "bash"

    @property
    def description(self) -> str:
        """Tool description for LLM.

        This description helps LLMs understand when and how to use
        this tool. It explains:
        - What the tool does
        - Required input format
        - Output format
        - Timeout behavior

        Returns:
            Human-readable description string.
        """
        return (
            "Execute bash shell commands. "
            "Input: command string to execute. "
            "Output: stdout, stderr, and return code. "
            f"Timeout: {self.timeout}s"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        """Command parameter schema.

        Defines the 'command' parameter that must be provided
        when calling execute().

        Returns:
            List with single ToolParameter for 'command'.
        """
        return [
            ToolParameter(
                name="command",
                type="string",
                description="The bash command to execute",
                required=True,
            )
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute bash command.

        Runs the command in a subprocess with configured timeout
        and working directory. Captures stdout and stderr.

        Args:
            **kwargs: Must include 'command' key with the command string.
                     Other keys are rejected by validation.

        Returns:
            ToolResult with:
            - success: True if command exited with code 0
            - data: BashResult with returncode, stdout, stderr
            - error: Error message if execution failed

        Raises:
            (No exceptions raised - all errors are captured in ToolResult)

        Example:
            >>> result = tool.execute(command="echo hello")
            >>> result.success
            True
            >>> result.data.stdout
            'hello\\n'
            >>>
            >>> # Failed command
            >>> result = tool.execute(command="cat /nonexistent")
            >>> result.success
            False
            >>> result.data.returncode
            1
        """
        # Validate arguments using parent class validation
        if error := self.validate(kwargs):
            return error

        command = kwargs["command"]

        try:
            # Run the command in subprocess
            result = subprocess.run(
                command,
                shell=True,  # Use shell to interpret command
                capture_output=True,  # Capture stdout/stderr
                text=True,  # Return strings not bytes
                timeout=self.timeout,  # Prevent hanging
                cwd=self.cwd,  # Working directory
            )

            # Package result in BashResult dataclass
            bash_result = BashResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

            # Return successful ToolResult with data
            return ToolResult.ok(bash_result)

        except subprocess.TimeoutExpired:
            # Command took too long
            return ToolResult.fail(f"Command timed out after {self.timeout} seconds")
        except Exception as e:
            # Any other error (permissions, etc.)
            return ToolResult.fail(f"Execution error: {e}")
