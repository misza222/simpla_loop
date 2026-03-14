"""Tests for core Tool abstractions."""

from conftest import DummyTool, NoParamsTool

from simpla_loop.core.tool import ToolParameter, ToolResult


class TestToolResult:
    """Test suite for ToolResult."""

    def test_ok_factory(self):
        """ok() should create successful result."""
        result = ToolResult.ok("data")

        assert result.success is True
        assert result.data == "data"
        assert result.error is None

    def test_fail_factory(self):
        """fail() should create failed result."""
        result = ToolResult.fail("error message")

        assert result.success is False
        assert result.data is None
        assert result.error == "error message"


class TestToolParameter:
    """Test suite for ToolParameter."""

    def test_required_defaults_true(self):
        """Required should default to True."""
        param = ToolParameter(name="test", type="string", description="test param")
        assert param.required is True

    def test_optional_with_default(self):
        """Optional parameter can have default."""
        param = ToolParameter(
            name="test",
            type="integer",
            description="test param",
            required=False,
            default=42,
        )
        assert param.required is False
        assert param.default == 42


class TestToolValidate:
    """Tests for Tool.validate() parameter validation."""

    def test_valid_required_only(self):
        """Providing only required params should pass validation."""
        tool = DummyTool()
        assert tool.validate({"required_param": "x"}) is None

    def test_valid_required_and_optional(self):
        """Providing both required and optional params should pass."""
        tool = DummyTool()
        assert tool.validate({"required_param": "x", "optional_param": 5}) is None

    def test_missing_required_param(self):
        """Omitting a required param should return a failure."""
        tool = DummyTool()
        result = tool.validate({"optional_param": 5})
        assert result is not None
        assert result.success is False
        assert "Missing required parameter" in result.error

    def test_unknown_param(self):
        """Providing an unknown param should return a failure."""
        tool = DummyTool()
        result = tool.validate({"required_param": "x", "bogus": 1})
        assert result is not None
        assert result.success is False
        assert "Unknown parameters" in result.error

    def test_no_params_tool_accepts_empty(self):
        """A tool with no parameters should accept an empty dict."""
        tool = NoParamsTool()
        assert tool.validate({}) is None

    def test_no_params_tool_rejects_unknown(self):
        """A tool with no parameters should reject any supplied param."""
        tool = NoParamsTool()
        result = tool.validate({"x": 1})
        assert result is not None
        assert result.success is False
        assert "Unknown parameters" in result.error
