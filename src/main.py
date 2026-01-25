from agent_core import Agent
from dotenv import load_dotenv
import os

def add(a: int, b: int) -> int:
    return a + b

load_dotenv()
agent = Agent(api_key=os.getenv("API_KEY"),
            model="claude-sonnet-4-20250514",
            max_iterations=5,)

agent.register_tool(
    name="add",
    description="Add two integers",
    func=add,
    parameters={
        "a": {"type": "integer", "required": True},
        "b": {"type": "integer", "required": True},
    },
)

result = agent.run("What is 12 + 30?", verbose=True)
print("\nFINAL RESULT:", result)