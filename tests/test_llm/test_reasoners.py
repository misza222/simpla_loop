"""Tests for reasoner factory functions."""

from unittest.mock import MagicMock, patch

from simpla_loop.core.tool import Tool, ToolParameter, ToolResult
from simpla_loop.llm.reasoners import (
    _build_prompt,
    _build_tools_description,
    create_react_reasoner,
)


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
        prompt = _build_prompt("What is 2+2?", [], [])

        assert "USER QUERY: What is 2+2?" in prompt
        assert "PREVIOUS STEPS" not in prompt

    def test_prompt_with_steps(self):
        """Should include step history."""
        from simpla_loop.loops.react import ReActStep

        steps = [
            ReActStep(
                thought="I should calculate",
                action="calculator",
                action_input={"expr": "2+2"},
                observation=ToolResult.ok("4"),
            )
        ]

        prompt = _build_prompt("Test query", steps, [])

        assert "Test query" in prompt
        assert "Step 1" in prompt
        assert "I should calculate" in prompt
        assert "calculator" in prompt


class TestCreateReactReasoner:
    """Test suite for create_react_reasoner."""

    @patch("simpla_loop.llm.reasoners.create_instructor_client")
    def test_reasoner_returns_dict(self, mock_create_client):
        """Reasoner should return expected dict format."""
        mock_response = MagicMock()
        mock_response.to_reasoner_dict.return_value = {
            "thought": "I will search",
            "action": "search",
            "action_input": {"query": "test"},
            "final_answer": None,
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_create_client.return_value = mock_client

        reasoner = create_react_reasoner(api_key="sk-test", model="gpt-4")
        result = reasoner("test query", [], [])

        assert result["thought"] == "I will search"
        assert result["action"] == "search"
        assert result["action_input"] == {"query": "test"}

    def test_config_override_priority(self, monkeypatch):
        """Explicit params should override env vars."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5")

        with patch("simpla_loop.llm.reasoners.create_instructor_client") as mock:
            mock.return_value = MagicMock()
            reasoner = create_react_reasoner(api_key="sk-explicit", model="gpt-4")

            assert callable(reasoner)
