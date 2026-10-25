"""AI Agent with Tools + Memory (LangChain + LangGraph + Groq).

Features:
- Tool calling: calculator, current date/time, Wikipedia lookup
- Conversation memory: remembers earlier messages in the same session
"""

import ast
import datetime
import operator
import os
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ---------- Tools ----------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    """Evaluate a math expression without using eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression such as '(12 + 8) * 3 / 4'."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"Could not calculate: {e}"


@tool
def current_datetime() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


@tool
def wikipedia_summary(topic: str) -> str:
    """Get a short Wikipedia summary for a topic, person, or concept."""
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(
        topic.replace(" ", "_")
    )
    try:
        r = requests.get(url, headers={"User-Agent": "ai-agent-demo/1.0"}, timeout=10)
        if r.status_code != 200:
            return f"No Wikipedia page found for '{topic}'."
        return r.json().get("extract", "No summary available.")
    except Exception as e:
        return f"Lookup failed: {e}"


TOOLS = [calculator, current_datetime, wikipedia_summary]

# ---------- Agent ----------

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Use the tools when they help: "
    "calculator for math, current_datetime for date/time questions, "
    "wikipedia_summary for factual lookups. Be concise and accurate."
)


def build_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    memory = MemorySaver()  # stores conversation history per thread_id
    return create_react_agent(llm, TOOLS, checkpointer=memory, prompt=SYSTEM_PROMPT)


def main():
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in your .env file first.")

    agent = build_agent()
    config = {"configurable": {"thread_id": "session-1"}}  # same id = same memory

    print("AI Agent ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        result = agent.invoke({"messages": [("user", user_input)]}, config)
        print(f"Agent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
