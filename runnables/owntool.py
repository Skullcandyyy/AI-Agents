from langchain.tools import tool

@tool #decorator to mark this function as a tool
def get_greeting(name: str) -> str: #type hints
    """Generate a greeting message for a user""" #docstring to describe the tool's purpose and input/output
    return f"Hello, {name}, welcome to the world of AI!"

result=get_greeting.invoke({"name": "Alice"})
print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)