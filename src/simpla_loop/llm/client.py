"""OpenAI client configuration and Instructor integration.

This module provides utilities for creating OpenAI clients with
Instructor patching for structured outputs. Configuration can be
loaded from environment variables or provided explicitly.
"""

import os
from dataclasses import dataclass
from typing import Any

import instructor
from openai import OpenAI


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI client.

    Configuration is loaded from environment variables by default,
    but can be overridden via constructor arguments.

    Environment Variables:
        OPENAI_API_KEY: Your OpenAI API key (required)
        OPENAI_BASE_URL: API base URL for custom gateways (optional)
        OPENAI_MODEL: Default model name (default: gpt-4o-mini)
        OPENAI_MAX_RETRIES: Max retries for instructor (default: 3)

    Attributes:
        api_key: OpenAI API key
        base_url: Optional custom API endpoint (for gateways)
        model: Model name to use
        max_retries: Maximum validation retries via instructor

    Example:
        >>> # From environment
        >>> config = OpenAIConfig.from_env()
        >>>
        >>> # With overrides
        >>> config = OpenAIConfig(
        ...     api_key="sk-...",  # pragma: allowlist secret
        ...     model="gpt-4",
        ...     base_url="https://gateway.example.com/v1"
        ... )
    """

    api_key: str
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    max_retries: int = 3

    @classmethod
    def from_env(cls, **overrides: Any) -> "OpenAIConfig":
        """Create config from environment variables.

        Reads configuration from environment variables:
        - OPENAI_API_KEY (required)
        - OPENAI_BASE_URL (optional)
        - OPENAI_MODEL (default: gpt-4o-mini)
        - OPENAI_MAX_RETRIES (default: 3)

        Args:
            **overrides: Keyword args to override env values

        Returns:
            OpenAIConfig with loaded values

        Raises:
            ValueError: If OPENAI_API_KEY is not set and not overridden

        Example:
            >>> import os
            >>> os.environ["OPENAI_API_KEY"] = "sk-test"  # pragma: allowlist secret
            >>> config = OpenAIConfig.from_env()
            >>> config.api_key
            'sk-test'
            >>>
            >>> # Override specific values
            >>> config = OpenAIConfig.from_env(model="gpt-4")
        """
        api_key = overrides.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Set it in .env file or pass api_key explicitly."
            )

        return cls(
            api_key=api_key,
            base_url=overrides.get("base_url") or os.getenv("OPENAI_BASE_URL"),
            model=overrides.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_retries=overrides.get("max_retries")
            or int(os.getenv("OPENAI_MAX_RETRIES", "3")),
        )


def create_instructor_client(
    config: OpenAIConfig | None = None,
) -> instructor.Instructor:
    """Create an Instructor-patched OpenAI client.

    Instructor patches the OpenAI client to provide structured
    output parsing with Pydantic models. It automatically:
    - Validates responses against Pydantic schemas
    - Retries on validation failures (up to max_retries)
    - Returns typed Pydantic objects instead of raw JSON

    Args:
        config: OpenAIConfig with connection settings.
               If None, loads from environment via OpenAIConfig.from_env()

    Returns:
        Instructor client ready for structured completions

    Example:
        >>> from simpla_loop.llm.models import ReActResponse
        >>>
        >>> config = OpenAIConfig.from_env()
        >>> client = create_instructor_client(config)
        >>>
        >>> # Use with structured output
        >>> response = client.chat.completions.create(
        ...     model=config.model,
        ...     messages=[{"role": "user", "content": "What is 2+2?"}],
        ...     response_model=ReActResponse,
        ... )
        >>> print(response.thought)
    """
    if config is None:
        config = OpenAIConfig.from_env()

    # Create base OpenAI client
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )

    # Patch with instructor for structured outputs
    return instructor.from_openai(
        client,
        mode=instructor.Mode.JSON,  # Use JSON mode for reliability
    )
