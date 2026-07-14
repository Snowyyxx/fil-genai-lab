from fastmcp import FastMCP
import math

mcp = FastMCP("Math_Solver_Tools")

@mcp.tool()
def calculate_expression(expression: str) -> str:
    """
    Evaluates basic mathematical expressions safely.
    Example input: "2 * (3 + 5) / math.sqrt(16)"
    """
    try:
        # Safe evaluation namespace containing standard math functions
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")