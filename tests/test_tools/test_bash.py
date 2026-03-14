"""Tests for BashTool."""

from simpla_loop.tools.bash import BashResult, BashTool


class TestBashTool:
    """Test suite for BashTool."""

    def test_name(self):
        """Tool name should be 'bash'."""
        tool = BashTool()
        assert tool.name == "bash"

    def test_description_includes_timeout(self):
        """Description should mention timeout."""
        tool = BashTool(timeout=30)
        assert "bash" in tool.description.lower()
        assert "30s" in tool.description

    def test_parameters(self):
        """Should have 'command' parameter."""
        tool = BashTool()
        params = tool.parameters

        assert len(params) == 1
        assert params[0].name == "command"
        assert params[0].required is True

    def test_execute_echo(self):
        """Should execute echo command."""
        tool = BashTool()
        result = tool.execute(command="echo 'hello world'")

        assert result.success is True
        assert isinstance(result.data, BashResult)
        assert result.data.returncode == 0
        assert "hello world" in result.data.stdout

    def test_execute_error_command(self):
        """Should handle failing commands."""
        tool = BashTool()
        result = tool.execute(command="exit 1")

        assert result.success is True  # Tool execution succeeded
        assert result.data.success is False  # But command failed
        assert result.data.returncode == 1

    def test_execute_missing_command_param(self):
        """Should fail without command parameter."""
        tool = BashTool()
        result = tool.execute()

        assert result.success is False
        assert "Missing required parameter" in result.error

    def test_execute_unknown_param(self):
        """Should fail with unknown parameter."""
        tool = BashTool()
        result = tool.execute(command="echo hi", unknown="value")

        assert result.success is False
        assert "Unknown parameters" in result.error

    def test_execute_timeout(self):
        """Should timeout long-running commands."""
        tool = BashTool(timeout=1)
        result = tool.execute(command="sleep 10")

        assert result.success is False
        assert "timed out" in result.error

    def test_execute_cwd(self):
        """Should respect working directory."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = BashTool(cwd=tmpdir)
            result = tool.execute(command="pwd")

            assert result.success is True
            assert tmpdir in result.data.stdout

    def test_bash_result_success_property(self):
        """BashResult.success should check returncode."""
        success_result = BashResult(returncode=0, stdout="", stderr="")
        fail_result = BashResult(returncode=1, stdout="", stderr="error")

        assert success_result.success is True
        assert fail_result.success is False
