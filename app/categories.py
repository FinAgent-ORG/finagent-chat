from __future__ import annotations

BUSINESS_CATEGORIES = (
    "Operational",
    "Inventory",
    "Employee",
    "Logistics",
    "Marketing",
    "Software",
    "Utilities",
    "Travel",
    "Compliance",
    "Miscellaneous",
)

DEFAULT_CATEGORY = "Miscellaneous"

LEGACY_CATEGORY_MAP = {
    "entertainment": "Marketing",
    "food": "Operational",
    "groceries": "Inventory",
    "healthcare": "Employee",
    "other": "Miscellaneous",
    "rent": "Operational",
    "transport": "Travel",
    "utilities": "Utilities",
}

CATEGORY_ALIASES = {
    "coffee": "Operational",
    "tea": "Operational",
    "cafe": "Operational",
    "restaurant": "Operational",
    "breakfast": "Operational",
    "lunch": "Operational",
    "dinner": "Operational",
    "snack": "Operational",
    "office": "Operational",
    "supplies": "Inventory",
    "inventory": "Inventory",
    "stock": "Inventory",
    "materials": "Inventory",
    "grocery": "Inventory",
    "groceries": "Inventory",
    "vegetables": "Inventory",
    "milk": "Inventory",
    "supermarket": "Inventory",
    "salary": "Employee",
    "payroll": "Employee",
    "reimbursement": "Employee",
    "benefits": "Employee",
    "doctor": "Employee",
    "hospital": "Employee",
    "medicine": "Employee",
    "pharmacy": "Employee",
    "shipping": "Logistics",
    "freight": "Logistics",
    "courier": "Logistics",
    "delivery": "Logistics",
    "warehouse": "Logistics",
    "fuel": "Logistics",
    "petrol": "Logistics",
    "diesel": "Logistics",
    "parking": "Logistics",
    "toll": "Logistics",
    "advertising": "Marketing",
    "ads": "Marketing",
    "campaign": "Marketing",
    "branding": "Marketing",
    "promotion": "Marketing",
    "movie": "Marketing",
    "netflix": "Marketing",
    "spotify": "Marketing",
    "game": "Marketing",
    "concert": "Marketing",
    "software": "Software",
    "saas": "Software",
    "license": "Software",
    "subscription": "Software",
    "hosting": "Software",
    "cloud": "Software",
    "electricity": "Utilities",
    "water": "Utilities",
    "internet": "Utilities",
    "wifi": "Utilities",
    "gas": "Utilities",
    "bill": "Utilities",
    "phone": "Utilities",
    "flight": "Travel",
    "airfare": "Travel",
    "hotel": "Travel",
    "lodging": "Travel",
    "taxi": "Travel",
    "uber": "Travel",
    "ola": "Travel",
    "bus": "Travel",
    "train": "Travel",
    "metro": "Travel",
    "cab": "Travel",
    "travel": "Travel",
    "legal": "Compliance",
    "audit": "Compliance",
    "tax": "Compliance",
    "filing": "Compliance",
    "insurance": "Compliance",
    "other": "Miscellaneous",
    "misc": "Miscellaneous",
}


def normalize_expense_category(value: str | None) -> str:
    if value is None:
        return DEFAULT_CATEGORY

    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        return DEFAULT_CATEGORY

    if cleaned in BUSINESS_CATEGORIES:
        return cleaned

    lowered = cleaned.lower()
    if lowered in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[lowered]

    if lowered in LEGACY_CATEGORY_MAP:
        return LEGACY_CATEGORY_MAP[lowered]

    return DEFAULT_CATEGORY
