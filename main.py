import os
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from agent import handle_chat
from schemas import ChatRequest, ChatResponse
from security import oauth2_scheme, require_user

load_dotenv()

app = FastAPI(title=os.getenv("APP_NAME", "finagent-chat-service"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

_request_log: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client_ip = (
        request.headers.get("x-forwarded-for")
        or request.headers.get("x-real-ip")
        or request.client.host
        or "anonymous"
    ).split(",")[0].strip()
    now = time.time()
    window = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    limit = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
    bucket = _request_log[client_ip]
    while bucket and now - bucket[0] > window:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
    bucket.append(now)
    return await call_next(request)


@app.get("/api/v1/chat/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/chat/messages", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    current_user: dict = Depends(require_user),
    token: str = Depends(oauth2_scheme),
) -> ChatResponse:
    try:
        response_text = await handle_chat(
            history=[item.model_dump() for item in payload.history],
            message=payload.message,
            token=token,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="AI chat agent failed.") from exc

    return ChatResponse(response=response_text.strip() or "I could not generate a response.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=os.getenv("APP_HOST", "0.0.0.0"), port=int(os.getenv("APP_PORT", "8004")), reload=True)
