"""Reasoner factory functions for LLM-based reasoning.

This module creates reasoner functions that use LLMs (via Instructor)
to generate structured reasoning outputs for agentic loops.
"""

import json
from typing import TYPE_CHECKING, Any

from simpla_loop.core.tool import Tool
from simpla_loop.llm.client import OpenAIConfig, create_instructor_client
from simpla_loop.llm.models import ReActResponse, ToolInfo

if TYPE_CHECKING:
    from simpla_loop.loops.react import ReActStep, Reasoner


# Default system prompt for ReAct reasoning
REACT_SYSTEM_PROMPT = """You are a helpful AI assistant that solves problems by thinking
step by step and using tools when needed.

You will be given:
1. A user query to answer
2. A history of previous thoughts, actions, and observations
3. A list of available tools

Your task is to decide what to do next. You should:
- Think about the problem and what information you need
- Use tools when you need external information or capabilities
- Provide a final answer when you have enough information

GUIDELINES:
- Be thorough in your reasoning - explain what you're thinking
- Use tools rather than guessing when you need facts
- If a tool fails, try a different approach
- When you provide a final answer, make sure it directly addresses the user's query

AVAILABLE TOOLS:
{tools_description}

RESPONSE FORMAT:
You must respond with a valid JSON object matching this schema:
{{
    "thought": "Your step-by-step reasoning. Explain what you understand.",
    "action": "tool_name_or_null",
    "action_input": {{"param": "value"}},
    "final_answer": "Your final answer if done, otherwise null"
}}

RULES:
- If you need to use a tool, set "action" to the tool name and provide "action_input"
- When done, set "final_answer" to your answer and "action" to null
- Always provide a thoughtful "thought" explaining your reasoning
- The "action_input" must match the tool's expected parameters
"""


def _build_tools_description(tools: list[Tool]) -> str:
    """Build a description string for available tools.

    Args:
        tools: List of available tools

    Returns:
        Formatted string describing all tools

    Example:
        >>> from simpla_loop.tools.bash import BashTool
        >>> desc = _build_tools_description([BashTool()])
        >>> "bash" in desc
        True
    """
    if not tools:
        return "No tools available. You must answer based on your knowledge."

    descriptions = []
    for tool in tools:
        info = ToolInfo.from_tool(tool)
        param_str = ""
        if info.parameters:
            param_lines = []
            for p in info.parameters:
                req = "required" if p["required"] else "optional"
                param_lines.append(
                    f"      - {p['name']} ({p['type']}): {p['description']} [{req}]"
                )
            param_str = "\n    Parameters:\n" + "\n".join(param_lines)

        desc = f"  - {info.name}: {info.description}{param_str}"
        descriptions.append(desc)

    return "\n".join(descriptions)


def _build_prompt(query: str, steps: list["ReActStep"], tools: list[Tool]) -> str:
    """Build the user prompt for the LLM.

    Args:
        query: The original user query
        steps: Previous steps in the conversation
        tools: Available tools

    Returns:
        Complete prompt string for the LLM
    """
    parts = [f"USER QUERY: {query}\n"]

    if steps:
        parts.append("PREVIOUS STEPS:")
        for i, step in enumerate(steps, 1):
            parts.append(f"\nStep {i}:")
            parts.append(f"  Thought: {step.thought}")
            if step.action:
                parts.append(f"  Action: {step.action}")
                parts.append(f"  Action Input: {json.dumps(step.action_input)}")
                if step.observation:
                    obs = step.observation
                    if hasattr(obs, "success"):
                        if obs.success:
                            parts.append(f"  Observation: {obs.data}")
                        else:
                            parts.append(f"  Observation (ERROR): {obs.error}")
                    else:
                        parts.append(f"  Observation: {obs}")
        parts.append("\n" + "=" * 50 + "\n")

    parts.append("What is your next thought and action? Respond with the JSON format.")

    return "\n".join(parts)


def create_react_reasoner(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_retries: int | None = None,
) -> "Reasoner":
    """Create a reasoner function for ReActLoop using an LLM.

    This factory function creates a reasoner that uses Instructor-patched
    OpenAI client to generate structured ReAct reasoning outputs. The
    reasoner handles:
    - Building prompts from query history
    - Calling the LLM with structured output requirements
    - Parsing and validating responses
    - Retrying on validation failures

    Configuration is loaded from environment variables by default:
    - OPENAI_API_KEY: Required API key
    - OPENAI_BASE_URL: Optional custom gateway
    - OPENAI_MODEL: Model name (default: gpt-4o-mini)
    - OPENAI_MAX_RETRIES: Retry count (default: 3)

    Args:
        model: Model name override (or from env OPENAI_MODEL)
        api_key: API key override (or from env OPENAI_API_KEY)
        base_url: Gateway URL override (or from env OPENAI_BASE_URL)
        max_retries: Retry count override (or from env OPENAI_MAX_RETRIES)
        system_prompt: Custom system prompt (or use default)

    Returns:
        A reasoner function compatible with ReActLoop

    Raises:
        ConfigError: If API key is not provided and not in environment

    Example:
        >>> # Using environment variables
        >>> reasoner = create_react_reasoner()
        >>>
        >>> # With explicit configuration
        >>> reasoner = create_react_reasoner(
        ...     model="gpt-4",
        ...     api_key="sk-...",  # pragma: allowlist secret
        ...     base_url="https://gateway.example.com/v1"
        ... )
        >>>
        >>> # Use in loop
        >>> loop = ReActLoop(reasoner=reasoner)

    Note:
        The returned reasoner has signature:
        (query: str, steps: list[ReActStep], tools: list[Tool]) -> dict
    """
    # Build config from env + overrides
    config_kwargs: dict[str, str | int] = {}
    if api_key is not None:
        config_kwargs["api_key"] = api_key
    if base_url is not None:
        config_kwargs["base_url"] = base_url
    if model is not None:
        config_kwargs["model"] = model
    if max_retries is not None:
        config_kwargs["max_retries"] = max_retries

    config = OpenAIConfig(**config_kwargs)  # type: ignore[arg-type]
    client = create_instructor_client(config)

    sys_prompt = REACT_SYSTEM_PROMPT

    def reasoner(
        query: str, steps: list["ReActStep"], tools: list[Tool]
    ) -> dict[str, Any]:
        """Generate the next reasoning step using LLM.

        Args:
            query: The original user query
            steps: History of previous steps (thoughts/actions/observations)
            tools: Available tools for this step

        Returns:
            Dict with keys: thought, action, action_input, final_answer
        """
        # Build tool descriptions
        tools_desc = _build_tools_description(tools)

        # Build messages
        system_msg = sys_prompt.format(tools_description=tools_desc)
        user_msg = _build_prompt(query, steps, tools)

        # Call LLM with structured output
        # Instructor handles retries and validation automatically
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_model=ReActResponse,
            max_retries=config.max_retries,
        )

        # Convert to dict format expected by ReActLoop
        result: dict[str, Any] = response.to_reasoner_dict()
        return result

    return reasoner
