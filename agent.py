import os
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict

from clients import create_expense, list_expenses, list_expenses_by_date
from prompts import SYSTEM_PROMPT_CHAT
from schemas import ExpenseCategory


def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
    )


@tool
async def save_expense(
    amount: float,
    category: ExpenseCategory,
    description: str,
    config: RunnableConfig,
) -> str:
    """Save a user's expense when the amount, category, and description are known."""
    token = config.get("configurable", {}).get("token")
    if not token:
        return "Unable to save expense because the user token is missing."

    expense = await create_expense(
        token=token,
        amount=amount,
        category=category.value,
        description=description.strip()[:500],
    )
    return (
        f"Saved expense {expense['amount']:.2f} {expense['currency']} "
        f"for {expense['description']} in category {expense['category']}."
    )


@tool
async def get_expenses(config: RunnableConfig) -> str:
    """Retrieve the current user's expenses."""
    token = config.get("configurable", {}).get("token")
    if not token:
        return "Unable to read expenses because the user token is missing."

    expenses = await list_expenses(token)
    if not expenses:
        return "No expenses found."

    return "\n".join(
        f"- {item['amount']} {item['currency']} | {item['category']} | {item['expense_date']} | {item['description']}"
        for item in expenses
    )


@tool
async def get_expenses_by_date(target_date: str, config: RunnableConfig) -> str:
    """Retrieve the current user's expenses for a date in YYYY-MM-DD format."""
    token = config.get("configurable", {}).get("token")
    if not token:
        return "Unable to read expenses because the user token is missing."

    try:
        expenses = await list_expenses_by_date(token, target_date)
    except ValueError:
        return "Date must be in YYYY-MM-DD format."

    if not expenses:
        return f"No expenses found on {target_date}."

    return "\n".join(
        f"- {item['amount']} {item['currency']} | {item['category']} | {item['expense_date']} | {item['description']}"
        for item in expenses
    )


TOOLS = [save_expense, get_expenses, get_expenses_by_date]
LLM_WITH_TOOLS = _build_llm().bind_tools(TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


async def reasoner_node(state: AgentState) -> dict:
    response = await LLM_WITH_TOOLS.ainvoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("reasoner", reasoner_node)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")

react_graph = builder.compile()


def build_messages(history: list[dict], message: str) -> list:
    messages: list = [SystemMessage(content=SYSTEM_PROMPT_CHAT)]
    for item in history:
        role = item.get("role")
        text = item.get("text", "")
        if role == "user":
            messages.append(HumanMessage(content=text))
        elif role == "assistant":
            messages.append(AIMessage(content=text))
    messages.append(HumanMessage(content=message))
    return messages
