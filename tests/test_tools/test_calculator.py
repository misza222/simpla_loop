"""Tests for CalculatorTool."""

from simpla_loop.tools.calculator import CalculatorTool


class TestCalculatorTool:
    """Test suite for CalculatorTool."""

    def test_name(self):
        """Tool should identify as 'calculator'."""
        tool = CalculatorTool()
        assert tool.name == "calculator"

    def test_description(self):
        """Description should mention arithmetic."""
        tool = CalculatorTool()
        assert "arithmetic" in tool.description.lower()

    def test_parameters(self):
        """Should have one required 'expression' parameter."""
        tool = CalculatorTool()
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "expression"
        assert tool.parameters[0].required is True

    def test_addition(self):
        """Should evaluate simple addition."""
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 3")
        assert result.success is True
        assert result.data == 5

    def test_multiplication_precedence(self):
        """Should respect operator precedence."""
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 3 * 4")
        assert result.success is True
        assert result.data == 14

    def test_parentheses(self):
        """Should handle parenthesized expressions."""
        tool = CalculatorTool()
        result = tool.execute(expression="(2 + 3) * 4")
        assert result.success is True
        assert result.data == 20

    def test_float_division(self):
        """Should handle float division."""
        tool = CalculatorTool()
        result = tool.execute(expression="7 / 2")
        assert result.success is True
        assert result.data == 3.5

    def test_floor_division(self):
        """Should handle floor division."""
        tool = CalculatorTool()
        result = tool.execute(expression="7 // 2")
        assert result.success is True
        assert result.data == 3

    def test_modulo(self):
        """Should handle modulo."""
        tool = CalculatorTool()
        result = tool.execute(expression="10 % 3")
        assert result.success is True
        assert result.data == 1

    def test_power(self):
        """Should handle exponentiation."""
        tool = CalculatorTool()
        result = tool.execute(expression="2 ** 10")
        assert result.success is True
        assert result.data == 1024

    def test_unary_negative(self):
        """Should handle unary minus."""
        tool = CalculatorTool()
        result = tool.execute(expression="-5 + 3")
        assert result.success is True
        assert result.data == -2

    def test_float_operands(self):
        """Should handle float literals."""
        tool = CalculatorTool()
        result = tool.execute(expression="1.5 + 2.5")
        assert result.success is True
        assert result.data == 4.0

    def test_division_by_zero(self):
        """Should return failure on division by zero."""
        tool = CalculatorTool()
        result = tool.execute(expression="1 / 0")
        assert result.success is False
        assert "Division by zero" in result.error

    def test_invalid_syntax(self):
        """Should return failure on malformed expression."""
        tool = CalculatorTool()
        result = tool.execute(expression="2 +")
        assert result.success is False
        assert "Invalid expression syntax" in result.error

    def test_rejects_function_calls(self):
        """Should reject function calls for safety."""
        tool = CalculatorTool()
        result = tool.execute(expression="__import__('os').system('ls')")
        assert result.success is False

    def test_rejects_variable_names(self):
        """Should reject variable references."""
        tool = CalculatorTool()
        result = tool.execute(expression="x + 1")
        assert result.success is False

    def test_missing_expression_param(self):
        """Should fail validation when expression is missing."""
        tool = CalculatorTool()
        result = tool.execute(foo="bar")
        assert result.success is False

    def test_unknown_param(self):
        """Should fail validation on unknown parameters."""
        tool = CalculatorTool()
        result = tool.execute(expression="1+1", extra="bad")
        assert result.success is False
