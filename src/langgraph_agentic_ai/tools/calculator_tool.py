import math
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Use for arithmetic, percentages, and math calculations.
    Examples: '2 + 2', 'sqrt(16)', '100 * 0.15', 'log(100)', '2 ** 10'
    """
    safe_dict = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "pow": pow,
        "floor": math.floor, "ceil": math.ceil,
        "factorial": math.factorial,
    }
    try:
        result = eval(expression, safe_dict)
        return f"Result: {result}"
    except Exception as ex:
        return f"Error evaluating '{expression}': {ex}"
