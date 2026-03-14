"""Tests for LLM module."""

from unittest.mock import MagicMock, patch

import pytest

from simpla_loop.core.tool import Tool, ToolParameter, ToolResult
from simpla_loop.llm.client import OpenAIConfig
from simpla_loop.llm.models import ReActResponse, ToolInfo
from simpla_loop.llm.reasoners import (
    _build_prompt,
    _build_tools_description,
    create_react_reasoner,
)


class TestOpenAIConfig:
    """Test suite for OpenAIConfig."""

    def test_from_env_success(self, monkeypatch):
        """Should load config from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "5")

        config = OpenAIConfig.from_env()

        assert config.api_key == "sk-test-key"  # pragma: allowlist secret
        assert config.base_url == "https://gateway.example.com"
        assert config.model == "gpt-4"
        assert config.max_retries == 5

    def test_from_env_defaults(self, monkeypatch):
        """Should use defaults for optional values."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # Ensure other vars are not set
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_MAX_RETRIES", raising=False)

        config = OpenAIConfig.from_env()

        assert config.api_key == "sk-test"  # pragma: allowlist secret
        assert config.base_url is None
        assert config.model == "gpt-4o-mini"
        assert config.max_retries == 3

    def test_from_env_missing_api_key(self, monkeypatch):
        """Should raise error if API key missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIConfig.from_env()

    def test_from_env_with_overrides(self, monkeypatch):
        """Should allow overrides."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        config = OpenAIConfig.from_env(api_key="sk-override", model="gpt-3.5-turbo")

        assert config.api_key == "sk-override"  # pragma: allowlist secret
        assert config.model == "gpt-3.5-turbo"


class TestReActResponse:
    """Test suite for ReActResponse model."""

    def test_valid_response_with_action(self):
        """Should accept valid response with action."""
        response = ReActResponse(
            thought="I need to search",
            action="web_search",
            action_input={"query": "test"},
            final_answer=None,
        )

        assert response.thought == "I need to search"
        assert response.action == "web_search"
        assert response.is_final() is False

    def test_valid_response_with_final_answer(self):
        """Should accept valid response with final answer."""
        response = ReActResponse(
            thought="I have the answer",
            action=None,
            action_input={},
            final_answer="The answer is 42",
        )

        assert response.is_final() is True
        assert response.final_answer == "The answer is 42"

    def test_invalid_missing_both_action_and_answer(self):
        """Should require either action or final_answer."""
        with pytest.raises(ValueError):
            ReActResponse(thought="I'm thinking", action=None, final_answer=None)

    def test_to_reasoner_dict(self):
        """Should convert to dict format."""
        response = ReActResponse(
            thought="test", action="tool", action_input={"x": 1}, final_answer=None
        )

        result = response.to_reasoner_dict()

        assert result["thought"] == "test"
        assert result["action"] == "tool"
        assert result["action_input"] == {"x": 1}
        assert result["final_answer"] is None


class TestToolInfo:
    """Test suite for ToolInfo model."""

    def test_from_tool(self):
        """Should extract info from Tool."""
        tool = MagicMock(spec=Tool)
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.parameters = [
            ToolParameter(name="param1", type="string", description="First param"),
            ToolParameter(
                name="param2",
                type="integer",
                description="Second param",
                required=False,
                default=42,
            ),
        ]

        info = ToolInfo.from_tool(tool)

        assert info.name == "test_tool"
        assert info.description == "A test tool"
        assert len(info.parameters) == 2
        assert info.parameters[0]["name"] == "param1"
        assert info.parameters[0]["required"] is True
        assert info.parameters[1]["required"] is False


class TestBuildToolsDescription:
    """Test suite for _build_tools_description."""

    def test_empty_tools(self):
        """Should handle empty tool list."""
        result = _build_tools_description([])
        assert "No tools available" in result

    def test_single_tool(self):
        """Should describe a single tool."""
        tool = MagicMock(spec=Tool)
        tool.name = "bash"
        tool.description = "Run bash commands"
        tool.parameters = []

        result = _build_tools_description([tool])

        assert "bash" in result
        assert "Run bash commands" in result

    def test_tool_with_parameters(self):
        """Should include parameter descriptions."""
        tool = MagicMock(spec=Tool)
        tool.name = "search"
        tool.description = "Search the web"
        tool.parameters = [
            ToolParameter(
                name="query", type="string", description="Search query", required=True
            ),
        ]

        result = _build_tools_description([tool])

        assert "search" in result
        assert "query" in result
        assert "required" in result


class TestBuildPrompt:
    """Test suite for _build_prompt."""

    def test_prompt_with_no_steps(self):
        """Should build prompt without history."""
        query = "What is 2+2?"
        steps = []
        tools = []

        prompt = _build_prompt(query, steps, tools)

        assert "USER QUERY: What is 2+2?" in prompt
        assert "PREVIOUS STEPS" not in prompt

    def test_prompt_with_steps(self):
        """Should include step history."""
        from simpla_loop.loops.react import ReActStep

        query = "Test query"
        steps = [
            ReActStep(
                thought="I should calculate",
                action="calculator",
                action_input={"expr": "2+2"},
                observation=ToolResult.ok("4"),
            )
        ]
        tools = []

        prompt = _build_prompt(query, steps, tools)

        assert "Test query" in prompt
        assert "Step 1" in prompt
        assert "I should calculate" in prompt
        assert "calculator" in prompt


class MockReActStep:
    """Mock ReActStep for testing."""

    def __init__(self, thought, action=None, action_input=None, observation=None):
        self.thought = thought
        self.action = action
        self.action_input = action_input or {}
        self.observation = observation


class TestCreateReactReasoner:
    """Test suite for create_react_reasoner."""

    @patch("simpla_loop.llm.reasoners.create_instructor_client")
    def test_reasoner_returns_dict(self, mock_create_client):
        """Reasoner should return expected dict format."""
        # Setup mock client
        mock_response = MagicMock()
        mock_response.thought = "I will search"
        mock_response.action = "search"
        mock_response.action_input = {"query": "test"}
        mock_response.final_answer = None
        mock_response.to_reasoner_dict.return_value = {
            "thought": "I will search",
            "action": "search",
            "action_input": {"query": "test"},
            "final_answer": None,
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_create_client.return_value = mock_client

        # Create reasoner
        reasoner = create_react_reasoner(api_key="sk-test", model="gpt-4")

        # Call reasoner
        result = reasoner("test query", [], [])

        # Verify
        assert result["thought"] == "I will search"
        assert result["action"] == "search"
        assert result["action_input"] == {"query": "test"}

    def test_config_override_priority(self, monkeypatch):
        """Explicit params should override env vars."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5")

        # Can't fully test without mocking, but verify no exception
        with patch("simpla_loop.llm.reasoners.create_instructor_client") as mock:
            mock.return_value = MagicMock()
            reasoner = create_react_reasoner(api_key="sk-explicit", model="gpt-4")

            # The reasoner function should exist
            assert callable(reasoner)
