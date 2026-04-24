import os
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def require_user(token: Annotated[str, Depends(oauth2_scheme)]) -> dict:
    try:
        payload = jwt.decode(
            token,
            os.getenv("JWT_SECRET_KEY", "development-secret"),
            algorithms=[os.getenv("JWT_ALGORITHM", "HS256")],
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.") from exc
    if not payload.get("sub"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Malformed token.")
    return payload
