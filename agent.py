import json
import os
import re
from typing import Iterable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from clients import create_expense, list_expenses, list_expenses_by_date
from prompts import SYSTEM_PROMPT_CHAT

CATEGORY_ALIASES = {
    "coffee": "Food",
    "breakfast": "Food",
    "lunch": "Food",
    "dinner": "Food",
    "snack": "Food",
    "taxi": "Transport",
    "uber": "Transport",
    "ola": "Transport",
    "bus": "Transport",
    "train": "Transport",
    "fuel": "Transport",
    "petrol": "Transport",
    "electricity": "Utilities",
    "water": "Utilities",
    "internet": "Utilities",
    "wifi": "Utilities",
    "movie": "Entertainment",
    "netflix": "Entertainment",
    "spotify": "Entertainment",
    "grocery": "Groceries",
    "groceries": "Groceries",
    "rent": "Rent",
    "doctor": "Healthcare",
    "hospital": "Healthcare",
    "medicine": "Healthcare",
}
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
HISTORY_KEYWORDS = (
    "expense",
    "expenses",
    "spent",
    "spend",
    "history",
    "ledger",
    "transactions",
    "transaction",
    "summary",
    "summarize",
    "show",
    "list",
    "recent",
    "month",
    "week",
    "today",
    "yesterday",
)
SAVE_HINTS = (
    "spent",
    "paid",
    "pay",
    "bought",
    "buy",
    "logged",
    "log",
    "add expense",
    "save expense",
    "record expense",
)


def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
    )


LLM = _build_llm()


def _clean_json_response(raw_text: str) -> str:
    trimmed = raw_text.strip()
    if trimmed.startswith("```"):
        lines = [line for line in trimmed.splitlines() if not line.startswith("```")]
        trimmed = "\n".join(lines).strip()
    return trimmed


def _extract_date(text: str) -> str | None:
    match = DATE_PATTERN.search(text)
    return match.group(0) if match else None


def _looks_like_history_request(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in HISTORY_KEYWORDS)


def _looks_like_save_request(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in SAVE_HINTS)


def _infer_category_from_text(message: str) -> str | None:
    lowered = message.lower()
    for alias, category in CATEGORY_ALIASES.items():
        if alias in lowered:
            return category
    return None


def _format_expenses(expenses: Iterable[dict]) -> str:
    lines = []
    for item in expenses:
        lines.append(
            f"- {item['amount']} {item['currency']} | {item['category']} | {item['expense_date']} | {item['description']}"
        )
    return "\n".join(lines)


async def _extract_expense_from_message(message: str) -> dict | None:
    prompt = (
        "You extract structured expense data from a user message.\n"
        "Return strict JSON only with this schema:\n"
        '{"amount": 12.5, "category": "Food", "description": "coffee", "currency": "INR", "missing_fields": []}\n'
        "Valid categories only: Food, Transport, Utilities, Entertainment, Groceries, Rent, Healthcare, Other.\n"
        "If amount is missing, set it to null and include \"amount\" in missing_fields.\n"
        "If description is missing, set a concise sensible description when possible; otherwise use null and include \"description\".\n"
        "Default currency to INR.\n"
        "Do not return markdown."
    )
    response = await LLM.ainvoke([SystemMessage(content=prompt), HumanMessage(content=message)])
    content = response.content if isinstance(response.content, str) else str(response.content)
    parsed = json.loads(_clean_json_response(content))

    category = parsed.get("category") or _infer_category_from_text(message) or "Other"
    missing_fields = [str(item) for item in parsed.get("missing_fields", []) if str(item).strip()]
    amount = parsed.get("amount")
    description = parsed.get("description")

    if amount in ("", None):
        if "amount" not in missing_fields:
            missing_fields.append("amount")
        amount = None

    if not description:
        if "description" not in missing_fields:
            missing_fields.append("description")

    return {
        "amount": amount,
        "category": category,
        "description": description,
        "currency": parsed.get("currency") or "INR",
        "missing_fields": missing_fields,
    }


async def _answer_with_context(history: list[dict], message: str, token: str) -> str:
    context = ""
    if _looks_like_history_request(message):
        try:
            target_date = _extract_date(message)
            expenses = await (list_expenses_by_date(token, target_date) if target_date else list_expenses(token, days=30))
        except Exception:
            expenses = []

        if expenses:
            context = (
                "Use the following real expense data when answering. Do not invent any records.\n"
                f"{_format_expenses(expenses)}"
            )
        else:
            context = "No matching expense records were found."

    messages: list = [SystemMessage(content=SYSTEM_PROMPT_CHAT)]
    if context:
        messages.append(SystemMessage(content=context))

    for item in history:
        if item.get("role") == "user":
            messages.append(HumanMessage(content=item.get("text", "")))
        elif item.get("role") == "assistant":
            messages.append(AIMessage(content=item.get("text", "")))
    messages.append(HumanMessage(content=message))

    response = await LLM.ainvoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content.strip() or "I could not generate a response."


async def handle_chat(history: list[dict], message: str, token: str) -> str:
    target_date = _extract_date(message)
    if target_date and _looks_like_history_request(message):
        expenses = await list_expenses_by_date(token, target_date)
        if not expenses:
            return f"No expenses found on {target_date}."
        return _format_expenses(expenses)

    if _looks_like_history_request(message) and not _looks_like_save_request(message):
        expenses = await list_expenses(token, days=30)
        if not expenses:
            return "No expenses found."
        return await _answer_with_context(history, message, token)

    if _looks_like_save_request(message):
        try:
            extracted = await _extract_expense_from_message(message)
        except Exception:
            extracted = None
        if not extracted:
            return "I could not understand that expense yet. Please include the amount and a short description."

        missing_fields = extracted["missing_fields"]
        if "amount" in missing_fields:
            return "I can save that expense once you tell me the amount."
        if "description" in missing_fields:
            return "I can save that expense once you add a short description."

        expense = await create_expense(
            token=token,
            amount=float(extracted["amount"]),
            category=extracted["category"],
            description=str(extracted["description"]).strip()[:500],
            currency=str(extracted["currency"] or "INR"),
        )
        return (
            f"Saved expense {expense['amount']:.2f} {expense['currency']} "
            f"for {expense['description']} in category {expense['category']}."
        )

    return await _answer_with_context(history, message, token)
