"""Abstract tool interface for agentic systems.

Tools are the agent's interface to the external world. They
enable the agent to perform actions beyond just reasoning.

The Tool abstraction provides:
1. Self-describing interface (name, description, parameters)
2. Validated execution with structured results
3. Extensibility for any external capability

Example:
    >>> tool = BashTool()
    >>> print(tool.name)
    'bash'
    >>> print(tool.description)
    'Execute bash shell commands...'
    >>> result = tool.execute(command="echo hello")
    >>> print(result.success)
    True
    >>> print(result.data.stdout)
    'hello\\n'
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolParameter:
    """Schema for a tool parameter.

    ToolParameter defines the interface for a single parameter
    that a tool accepts. This schema is used for:
    - Documentation (what does this parameter do?)
    - Validation (checking arguments before execution)
    - LLM prompting (helping LLMs generate correct calls)

    Attributes:
        name: Parameter identifier. Must be a valid Python identifier.
              Used as the keyword argument name in execute().
        type: String describing the expected type.
              Common values: "string", "integer", "boolean", "array", "object"
              This is informational; actual validation is implementation-specific.
        description: Human-readable explanation of the parameter's purpose.
                     Important for LLM-based agents to use tools correctly.
        required: Whether this parameter must be provided.
                  If False and not provided, default is used.
        default: Value to use if parameter is not provided.
                 Only meaningful if required=False.

    Example:
        >>> param = ToolParameter(
        ...     name="command",
        ...     type="string",
        ...     description="The bash command to execute",
        ...     required=True
        ... )
        >>> param = ToolParameter(
        ...     name="timeout",
        ...     type="integer",
        ...     description="Seconds to wait",
        ...     required=False,
        ...     default=30
        ... )
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolResult:
    """Result of a tool execution.

    ToolResult provides a uniform way for tools to report:
    - Whether execution succeeded
    - What data was produced
    - What went wrong (if anything)

    This structure allows loops to handle tool outputs consistently,
    regardless of what specific tool was called.

    Attributes:
        success: True if the tool executed successfully and produced
                 valid output. False if execution failed.
        data: The output from the tool. Can be any type.
              Meaningful only if success=True.
              For failed executions, this is typically None.
        error: Error message if success=False.
               Should be a human-readable string explaining what failed.
               None if success=True.

    Class Methods:
        ok: Factory method to create a successful result
        fail: Factory method to create a failed result

    Example:
        >>> # Successful execution
        >>> result = ToolResult.ok(data={"files": ["a.txt", "b.txt"]})
        >>>
        >>> # Failed execution
        >>> result = ToolResult.fail("File not found: /tmp/data.txt")
    """

    success: bool
    data: Any
    error: str | None = None

    @classmethod
    def ok(cls, data: Any) -> "ToolResult":
        """Create a successful result.

        Args:
            data: The output data from the tool execution.

        Returns:
            ToolResult with success=True and the provided data.

        Example:
            >>> result = ToolResult.ok("Command output here")
            >>> result.success
            True
        """
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        """Create a failed result.

        Args:
            error: Human-readable error message.

        Returns:
            ToolResult with success=False and the error message.

        Example:
            >>> result = ToolResult.fail("Permission denied")
            >>> result.success
            False
            >>> result.error
            'Permission denied'
        """
        return cls(success=False, data=None, error=error)


class Tool(ABC):
    """Abstract interface for agent tools.

    A Tool represents a capability the agent can invoke.
    Each tool has:
    - A name (how the agent references it)
    - A description (what it does, for LLM prompting)
    - A parameter schema (what inputs it needs)
    - An execution method (what it actually does)

    Tools are designed to be:
    1. Self-describing: Tools define their own interface
    2. Stateless: No persistent state between calls
    3. Safe: Execution is sandboxed and validated
    4. Observable: Results are structured and logged

    Subclassing:
    When creating a new tool, override:
    - name (property): Return the tool identifier
    - description (property): Return what the tool does
    - execute (method): Implement the tool's behavior
    Optionally override:
    - parameters (property): Define parameter schema
    - validate (method): Custom validation logic

    Attributes:
        None (tools are defined by their methods/properties)

    Example:
        >>> class GreetingTool(Tool):
        ...     @property
        ...     def name(self) -> str:
        ...         return "greet"
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Generate a greeting message"
        ...
        ...     @property
        ...     def parameters(self) -> list[ToolParameter]:
        ...         return [
        ...             ToolParameter(
        ...                 name="name",
        ...                 type="string",
        ...                 description="Name to greet",
        ...                 required=True
        ...             )
        ...         ]
        ...
        ...     def execute(self, **kwargs) -> ToolResult:
        ...         name = kwargs.get("name", "World")
        ...         return ToolResult.ok(f"Hello, {name}!")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool.

        This name is used by:
        - The agent to reference the tool
        - Loops to look up tools by name
        - LLMs to decide which tool to call
        - Logging and debugging

        Requirements:
        - Must be unique within an agent's tool set
        - Should be descriptive and concise
        - Use snake_case convention
        - No spaces or special characters

        Returns:
            String identifier for this tool.

        Example:
            >>> tool.name
            'bash'
            >>> tool.name
            'web_search'
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does.

        This description is crucial for LLM-based agents to understand
        when and how to use the tool. It should explain:
        - What the tool does
        - When to use it
        - What inputs it needs
        - What output to expect

        Keep it concise but informative. LLMs have context limits.

        Returns:
            Description string for this tool.

        Example:
            >>> tool.description
            'Execute bash shell commands. Input: command string. Output: stdout/stderr.'
        """
        ...

    @property
    def parameters(self) -> list[ToolParameter]:
        """Schema describing expected parameters.

        This schema is used for:
        - Validating arguments before execution
        - Generating documentation
        - Prompting LLMs with tool signatures

        Default implementation returns an empty list (no parameters required).
        Override this if your tool accepts parameters.

        Returns:
            List of ToolParameter objects describing each parameter.

        Example:
            >>> tool.parameters
            [ToolParameter(name='command', type='string', description='...',
            required=True)]
        """
        return []

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments.

        This is the core method that performs the tool's action.
        Implementations should:
        1. Validate arguments (or rely on validate() being called first)
        2. Perform the tool's operation
        3. Return a ToolResult indicating success/failure

        The method should be side-effect-free in terms of the tool object
        itself (tools are stateless), but may have external side effects
        (file operations, network calls, etc.).

        Args:
            **kwargs: Named arguments matching the parameter schema.
                      Keys are parameter names, values are the arguments.

        Returns:
            ToolResult containing success status and output data.

        Raises:
            Exception: Tools may raise exceptions, but it's better to
                      catch errors and return ToolResult.fail() for
                      expected failure modes.

        Example:
            >>> result = tool.execute(command="ls -la")
            >>> if result.success:
            ...     print(result.data.stdout)
            ... else:
            ...     print(f"Error: {result.error}")
        """
        ...

    def validate(self, kwargs: dict[str, Any]) -> ToolResult | None:
        """Validate arguments against parameter schema.

        This method checks that provided arguments match the tool's
        parameter schema. It validates:
        - No unknown parameters are provided
        - All required parameters are present

        Subclasses may override this to add type checking or other
        validation logic.

        Args:
            kwargs: Arguments to validate (dictionary from execute()).

        Returns:
            None if valid, ToolResult.fail() if invalid.

        Example:
            >>> tool.validate({"command": "ls"})  # Valid
            None
            >>> tool.validate({"unknown": "value"})  # Invalid
            ToolResult(success=False, data=None, error="Unknown parameters: ...")
        """
        param_names = {p.name for p in self.parameters}

        # Check for unknown parameters
        unknown = set(kwargs.keys()) - param_names
        if unknown:
            return ToolResult.fail(f"Unknown parameters: {unknown}")

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return ToolResult.fail(f"Missing required parameter: {param.name}")

        return None
