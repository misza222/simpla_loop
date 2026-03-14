"""Calculator tool for evaluating arithmetic expressions.

Provides safe math evaluation without using eval() or exec().
Uses Python's ast module to parse and evaluate only arithmetic
operations, preventing code injection.

Supported operations: +, -, *, /, //, %, ** and unary +/-.
Operands can be integers or floats.

Example:
    >>> tool = CalculatorTool()
    >>> result = tool.execute(expression="2 + 3 * 4")
    >>> result.data
    14
"""

import ast
import operator
from typing import Any

from simpla_loop.core.tool import Tool, ToolParameter, ToolResult

# Mapping from AST node types to Python operators.
# Only arithmetic operations are allowed — no bitwise, no comparisons.
_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> int | float:
    """Recursively evaluate an AST node containing only arithmetic.

    Raises:
        ValueError: If the expression contains unsupported operations.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))

    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.left), _safe_eval(node.right))

    raise ValueError(
        f"Unsupported expression element: {type(node).__name__}. "
        "Only arithmetic with numbers is allowed."
    )


class CalculatorTool(Tool):
    """Evaluate arithmetic expressions safely.

    Uses Python's ast module to parse expressions into an AST and
    evaluates only numeric literals and arithmetic operators. This
    prevents any form of code injection — no function calls, attribute
    access, or variable references are permitted.

    Example:
        >>> tool = CalculatorTool()
        >>> tool.execute(expression="(10 + 5) * 2").data
        30
        >>> tool.execute(expression="7 / 0").success
        False
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate arithmetic expressions. "
            "Supports +, -, *, /, //, %, ** with integers and floats."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="Arithmetic expression to evaluate (e.g. '2 + 3 * 4')",
                required=True,
            ),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Evaluate the arithmetic expression.

        Args:
            **kwargs: Must contain 'expression' (str).

        Returns:
            ToolResult with the numeric result, or a failure message
            if the expression is invalid or causes an arithmetic error.
        """
        if error := self.validate(kwargs):
            return error

        expression = kwargs["expression"]

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return ToolResult.fail(f"Invalid expression syntax: {expression}")

        try:
            result = _safe_eval(tree)
        except ZeroDivisionError:
            return ToolResult.fail("Division by zero")
        except ValueError as exc:
            return ToolResult.fail(str(exc))

        return ToolResult.ok(result)
