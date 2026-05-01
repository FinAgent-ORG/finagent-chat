from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return " ".join(value.strip().split())


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        return " ".join(value.strip().split())


class ChatResponse(BaseModel):
    response: str


class ExpenseCategory(str, Enum):
    OPERATIONAL = "Operational"
    INVENTORY = "Inventory"
    EMPLOYEE = "Employee"
    LOGISTICS = "Logistics"
    MARKETING = "Marketing"
    SOFTWARE = "Software"
    UTILITIES = "Utilities"
    TRAVEL = "Travel"
    COMPLIANCE = "Compliance"
    MISCELLANEOUS = "Miscellaneous"
