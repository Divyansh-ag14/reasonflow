import sys
import io
from langchain_core.tools import tool


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the output. Input must be valid Python. Use print() to display results."""
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(code, {"__builtins__": __builtins__})
        output = redirected_output.getvalue()
        return output if output else "Code executed successfully (no output)."
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout


def get_python_repl_tool():
    """Returns a Python REPL tool for writing and executing Python code."""
    return python_repl
