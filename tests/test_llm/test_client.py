"""Tests for OpenAIConfig."""

import pytest

from simpla_loop.core.exceptions import ConfigError
from simpla_loop.llm.client import OpenAIConfig


class TestOpenAIConfig:
    """Test suite for OpenAIConfig."""

    def test_from_env_success(self, monkeypatch):
        """Should load config from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "5")

        config = OpenAIConfig()

        assert config.api_key == "sk-test-key"  # pragma: allowlist secret
        assert config.base_url == "https://gateway.example.com"
        assert config.model == "gpt-4"
        assert config.max_retries == 5

    def test_from_env_defaults(self, monkeypatch):
        """Should use defaults for optional values."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_MAX_RETRIES", raising=False)

        # _env_file=None prevents pydantic-settings from reading the real .env file
        config = OpenAIConfig(_env_file=None)

        assert config.api_key == "sk-test"  # pragma: allowlist secret
        assert config.base_url is None
        assert config.model == "gpt-4o-mini"
        assert config.max_retries == 3

    def test_missing_api_key_raises_config_error(self, monkeypatch):
        """Should raise ConfigError if API key missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # _env_file=None prevents pydantic-settings from reading the real .env file
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            OpenAIConfig(_env_file=None)

    def test_constructor_overrides_env(self, monkeypatch):
        """Explicit constructor args should override env vars."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        config = OpenAIConfig(api_key="sk-override", model="gpt-3.5-turbo")

        assert config.api_key == "sk-override"  # pragma: allowlist secret
        assert config.model == "gpt-3.5-turbo"
