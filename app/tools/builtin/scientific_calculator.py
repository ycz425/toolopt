import ast
import math
import operator

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class ScientificCalculatorResult(BaseModel):
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

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        return _FUNCTIONS[node.func.id](*(_eval_node(a) for a in node.args))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def scientific_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


class ScientificCalculatorTool(Tool):
    metadata = ToolMetadata(
        name="scientific_calculator",
        description=(
            "Evaluate an arithmetic expression that may include scientific functions "
            "(sqrt, sin, cos, tan, log, log10, exp) in addition to basic operators -- "
            "more powerful (and costlier) than the basic calculator."
        ),
        parameters=[
            ToolParameter(name="expression", type="string", description="A math expression, e.g. 'sqrt(16) + 3'."),
        ],
        returns=ScientificCalculatorResult.__name__,
    )

    def execute(self, expression: str) -> ScientificCalculatorResult:
        return ScientificCalculatorResult(result=round(scientific_eval(expression), 6))

    def cost(self, expression: str) -> float:
        return 0.03
