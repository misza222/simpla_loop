"""OpenAI client configuration and Instructor integration.

This module provides utilities for creating OpenAI clients with
Instructor patching for structured outputs. Configuration is loaded
automatically from environment variables or can be overridden via
constructor arguments.
"""

import instructor  # type: ignore[import-untyped]
from openai import OpenAI
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from simpla_loop.core.exceptions import ConfigError


class OpenAIConfig(BaseSettings):
    """Configuration for OpenAI client.

    Reads configuration from environment variables automatically.
    Any field can be overridden by passing it as a constructor argument.

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
        >>> config = OpenAIConfig()
        >>>
        >>> # With overrides
        >>> config = OpenAIConfig(
        ...     api_key="sk-...",  # pragma: allowlist secret
        ...     model="gpt-4",
        ...     base_url="https://gateway.example.com/v1"
        ... )
    """

    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env")

    api_key: str = ""
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    max_retries: int = 3

    @model_validator(mode="after")
    def check_api_key(self) -> "OpenAIConfig":
        """Raise ConfigError early if api_key is missing."""
        if not self.api_key:
            raise ConfigError(
                "OPENAI_API_KEY not found in environment. "
                "Set it in .env file or pass api_key explicitly."
            )
        return self


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
               If None, loads from environment via OpenAIConfig()

    Returns:
        Instructor client ready for structured completions

    Example:
        >>> from simpla_loop.llm.models import ReActResponse
        >>>
        >>> config = OpenAIConfig()
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
        config = OpenAIConfig()

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
