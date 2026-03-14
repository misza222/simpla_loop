"""Tests for LLM Pydantic models."""

from unittest.mock import MagicMock

import pytest

from simpla_loop.core.tool import Tool, ToolParameter
from simpla_loop.llm.models import ReActResponse, ToolInfo


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
