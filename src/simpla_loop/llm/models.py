"""Pydantic models for LLM structured outputs.

These models define the schema for LLM responses and provide
validation via Pydantic. Instructor uses these to ensure the
LLM returns properly structured JSON.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from simpla_loop.core.tool import Tool


class ReActResponse(BaseModel):
    """Structured output for ReAct reasoning step.

    This model defines the expected JSON structure from the LLM
    during each step of the ReAct loop. Instructor validates
    responses against this schema and retries if invalid.

    Attributes:
        thought: The agent's reasoning about what to do next.
                Explains the reasoning process for transparency.
        action: Name of the tool to call, or None if providing
                final answer. Must match a tool name in the
                available tools list.
        action_input: Arguments to pass to the tool, as a dict.
                     Must match the tool's parameter schema.
        final_answer: The final answer if done, otherwise None.
                     When provided, action should be None.

    Example:
        >>> response = ReActResponse(
        ...     thought="I need to search for information",
        ...     action="web_search",
        ...     action_input={"query": "Python programming"},
        ...     final_answer=None
        ... )
        >>> response.thought
        'I need to search for information'

        >>> # Final answer
        >>> response = ReActResponse(
        ...     thought="I have found the answer",
        ...     action=None,
        ...     action_input={},
        ...     final_answer="The answer is 42"
        ... )
    """

    thought: str = Field(
        ...,
        description=(
            "Your reasoning about what to do next."
            " Explain your thinking process clearly."
        ),
    )
    action: str | None = Field(
        None,
        description=(
            "The name of the tool to use, or null if you're providing the final answer"
        ),
    )
    action_input: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The input parameters for the tool, as a JSON object. Empty if no action."
        ),
    )
    final_answer: str | None = Field(
        None,
        description=(
            "Your final answer to the user's question."
            " Only provide this when you're done."
        ),
    )

    @model_validator(mode="after")
    def validate_action_or_answer(self) -> "ReActResponse":
        """Ensure either action or final_answer is provided."""
        if self.action is None and self.final_answer is None:
            raise ValueError("Either 'action' or 'final_answer' must be provided")
        return self

    @field_validator("action_input")
    @classmethod
    def validate_action_input(cls, action_input: dict, info) -> dict:
        """Ensure action_input is provided when action is set."""
        values = info.data
        action = values.get("action")

        if action is not None and not isinstance(action_input, dict):
            raise ValueError("action_input must be a dict when action is provided")

        return action_input

    def is_final(self) -> bool:
        """Check if this response provides a final answer.

        Returns:
            True if final_answer is set, False otherwise.

        Example:
            >>> r = ReActResponse(thought="done", final_answer="42")
            >>> r.is_final()
            True
        """
        return self.final_answer is not None

    def to_reasoner_dict(self) -> dict:
        """Convert to dict format expected by ReActLoop.

        Returns:
            Dict with keys: thought, action, action_input, final_answer

        Example:
            >>> r = ReActResponse(thought="test", action="tool", action_input={"x": 1})
            >>> r.to_reasoner_dict()
            {'thought': 'test', 'action': 'tool', 'action_input': {'x': 1},
            'final_answer': None}
        """
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "final_answer": self.final_answer,
        }


class ToolInfo(BaseModel):
    """Information about an available tool for the prompt.

        Used to build the system prompt describing available tools
    to the LLM.
    """

    name: str
    description: str
    parameters: list[dict]

    @classmethod
    def from_tool(cls, tool: Tool) -> "ToolInfo":
        """Create ToolInfo from a Tool instance.

        Args:
            tool: The tool to describe

        Returns:
            ToolInfo with extracted metadata

        Example:
            >>> from simpla_loop.tools.bash import BashTool
            >>> info = ToolInfo.from_tool(BashTool())
            >>> info.name
            'bash'
        """
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=[
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in tool.parameters
            ],
        )
