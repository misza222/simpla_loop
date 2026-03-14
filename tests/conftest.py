"""Pytest configuration and fixtures."""

import sys
from pathlib import Path
from typing import Any

import pytest

from simpla_loop.core.tool import Tool, ToolParameter, ToolResult

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class MockTool(Tool):
    """Mock tool for testing — configurable name and result."""

    def __init__(self, name: str, result: Any = None):
        self._name = name
        self._result = result or "mock_result"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Mock tool {self._name}"

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult.ok(self._result)


class DummyTool(Tool):
    """Dummy tool with one required and one optional parameter for validation tests."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool for testing"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="required_param",
                type="string",
                description="A required param",
                required=True,
            ),
            ToolParameter(
                name="optional_param",
                type="integer",
                description="An optional param",
                required=False,
                default=0,
            ),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult.ok("done")


class NoParamsTool(Tool):
    """Tool with zero parameters for validation edge-case tests."""

    @property
    def name(self) -> str:
        return "no_params"

    @property
    def description(self) -> str:
        return "Tool with no parameters"

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult.ok("ok")


@pytest.fixture
def mock_tool() -> MockTool:
    return MockTool("mock_tool")


@pytest.fixture
def dummy_tool() -> DummyTool:
    return DummyTool()


@pytest.fixture
def no_params_tool() -> NoParamsTool:
    return NoParamsTool()
