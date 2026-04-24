import os
from datetime import datetime

import httpx


async def create_expense(token: str, amount: float, category: str, description: str, currency: str = "INR") -> dict:
    async with httpx.AsyncClient(base_url=os.getenv("EXPENSE_SERVICE_BASE_URL", "http://localhost:8002"), timeout=20.0) as client:
        response = await client.post(
            "/api/v1/expenses",
            json={
                "amount": amount,
                "currency": currency,
                "category": category,
                "description": description,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()


async def list_expenses(token: str, days: int | None = None) -> list[dict]:
    params = {"days": days} if days else None
    async with httpx.AsyncClient(base_url=os.getenv("EXPENSE_SERVICE_BASE_URL", "http://localhost:8002"), timeout=20.0) as client:
        response = await client.get(
            "/api/v1/expenses",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()


async def list_expenses_by_date(token: str, target_date: str) -> list[dict]:
    parsed = datetime.strptime(target_date, "%Y-%m-%d").date()
    expenses = await list_expenses(token)
    return [expense for expense in expenses if expense.get("expense_date") == parsed.isoformat()]
