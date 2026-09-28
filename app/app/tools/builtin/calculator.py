import ast
import operator

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class CalculatorResult(BaseModel):
    result: float = Field(description="The numeric result of evaluating the expression.")


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


class CalculatorTool(Tool):
    metadata = ToolMetadata(
        name="calculator",
        description="Evaluates a basic arithmetic expression (+, -, *, /, **, parentheses) and returns the numeric result.",
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="A math expression to evaluate, e.g. '3 + 4 * 2'.",
            )
        ],
        returns=CalculatorResult.__name__,
    )

    def execute(self, expression: str) -> CalculatorResult:
        return CalculatorResult(result=safe_eval(expression))

    def cost(self, expression: str) -> float:
        return 0.01
