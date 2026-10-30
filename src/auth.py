"""
Authentication module for DeepDetect API.

Provides:
  - Password hashing with bcrypt
  - JWT access + refresh token issuance / verification
  - FastAPI dependency for protected routes (get_current_user)
  - SQLite-backed user store (same DB as jobs via database.py)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (override via environment variables)
# ---------------------------------------------------------------------------

SECRET_KEY: str = os.getenv(
    "DEEPDETECT_SECRET", "change-me-in-production-please-32chars!"
)
ACCESS_TOKEN_EXPIRE_SECS: int = int(os.getenv("ACCESS_EXPIRE_SECS", "3600"))   # 1 h
REFRESH_TOKEN_EXPIRE_SECS: int = int(os.getenv("REFRESH_EXPIRE_SECS", "604800"))  # 7 d

# ---------------------------------------------------------------------------
# JWT (pure-stdlib, no PyJWT dependency)
# ---------------------------------------------------------------------------

import base64
import json as _json


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))


def _jwt_sign(payload: dict) -> str:
    header = _b64url_encode(_json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64url_encode(_json.dumps(payload).encode())
    msg = f"{header}.{body}".encode()
    sig = hmac.new(SECRET_KEY.encode(), msg, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url_encode(sig)}"


def _jwt_verify(token: str) -> dict:
    try:
        header_b64, body_b64, sig_b64 = token.split(".")
    except ValueError:
        raise ValueError("Malformed token.")
    msg = f"{header_b64}.{body_b64}".encode()
    expected_sig = hmac.new(SECRET_KEY.encode(), msg, hashlib.sha256).digest()
    actual_sig = _b64url_decode(sig_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid token signature.")
    payload = _json.loads(_b64url_decode(body_b64))
    if payload.get("exp", 0) < time.time():
        raise ValueError("Token expired.")
    return payload


# ---------------------------------------------------------------------------
# Password hashing (bcrypt via passlib, fallback to PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------

try:
    import bcrypt as _bcrypt

    def hash_password(plain: str) -> str:
        pw_bytes = plain.encode("utf-8")[:72]
        salt = _bcrypt.gensalt()
        return _bcrypt.hashpw(pw_bytes, salt).decode("utf-8")

    def verify_password(plain: str, hashed: str) -> bool:
        try:
            pw_bytes = plain.encode("utf-8")[:72]
            return _bcrypt.checkpw(pw_bytes, hashed.encode("utf-8"))
        except Exception:
            return False

except ImportError:
    import hashlib as _hl

    def hash_password(plain: str) -> str:  # type: ignore[misc]
        salt = os.urandom(16).hex()
        dk = _hl.pbkdf2_hmac("sha256", plain.encode(), salt.encode(), 260000)
        return f"pbkdf2${salt}${dk.hex()}"

    def verify_password(plain: str, hashed: str) -> bool:  # type: ignore[misc]
        try:
            _, salt, stored = hashed.split("$", 2)
            dk = _hl.pbkdf2_hmac("sha256", plain.encode(), salt.encode(), 260000)
            return hmac.compare_digest(dk.hex(), stored)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

DB_PATH: str = os.getenv(
    "DEEPDETECT_DB", os.path.join(os.path.dirname(__file__), "..", "deepdetect.db")
)


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_auth_tables() -> None:
    """Create users and refresh_tokens tables if missing."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                email       TEXT UNIQUE NOT NULL,
                username    TEXT UNIQUE NOT NULL,
                hashed_pw   TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                is_active   INTEGER NOT NULL DEFAULT 1,
                reset_token TEXT,
                reset_expiry REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                token       TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                expires_at  REAL NOT NULL
            )
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class UserRegisterRequest(BaseModel):
    email: str
    username: str
    password: str


class UserLoginRequest(BaseModel):
    email: str
    password: str


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


class RefreshRequest(BaseModel):
    refresh_token: str


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def create_user(email: str, username: str, password: str) -> dict:
    """Insert a new user and return their record."""
    user_id = uuid.uuid4().hex
    hashed = hash_password(password)
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (id, email, username, hashed_pw, created_at) VALUES (?,?,?,?,?)",
                (user_id, email.lower(), username, hashed, now),
            )
            conn.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("Email or username already registered.") from exc
    return {"id": user_id, "email": email, "username": username, "created_at": now}


def get_user_by_email(email: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email=?", (email.lower(),)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def issue_tokens(user: dict) -> TokenResponse:
    now = time.time()
    access_payload = {
        "sub": user["id"],
        "username": user["username"],
        "iat": now,
        "exp": now + ACCESS_TOKEN_EXPIRE_SECS,
        "type": "access",
    }
    access = _jwt_sign(access_payload)

    refresh_token = uuid.uuid4().hex
    expires_at = now + REFRESH_TOKEN_EXPIRE_SECS
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO refresh_tokens (token, user_id, expires_at) VALUES (?,?,?)",
            (refresh_token, user["id"], expires_at),
        )
        conn.commit()

    return TokenResponse(
        access_token=access,
        refresh_token=refresh_token,
        user_id=user["id"],
        username=user["username"],
    )


def redeem_refresh_token(refresh_token: str) -> TokenResponse:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM refresh_tokens WHERE token=?", (refresh_token,)
        ).fetchone()
        if not row or row["expires_at"] < time.time():
            raise ValueError("Refresh token invalid or expired.")
        # Rotate: delete old, issue new
        conn.execute("DELETE FROM refresh_tokens WHERE token=?", (refresh_token,))
        conn.commit()

    user = get_user_by_id(row["user_id"])
    if not user or not user["is_active"]:
        raise ValueError("User account not found or disabled.")
    return issue_tokens(user)


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------

def create_password_reset_token(email: str) -> str:
    """
    Generate a reset token (valid 1 hour).
    In production, email this to the user; here we return it directly for
    local/demo use.
    """
    user = get_user_by_email(email)
    if not user:
        # Don't reveal whether the email exists
        return ""
    reset_token = uuid.uuid4().hex
    expiry = time.time() + 3600
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET reset_token=?, reset_expiry=? WHERE id=?",
            (reset_token, expiry, user["id"]),
        )
        conn.commit()
    return reset_token


def confirm_password_reset(token: str, new_password: str) -> bool:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, reset_expiry FROM users WHERE reset_token=?", (token,)
        ).fetchone()
        if not row or row["reset_expiry"] < time.time():
            return False
        hashed = hash_password(new_password)
        conn.execute(
            "UPDATE users SET hashed_pw=?, reset_token=NULL, reset_expiry=NULL WHERE id=?",
            (hashed, row["id"]),
        )
        conn.commit()
    return True


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> dict:
    """Dependency: validate JWT and return the user dict."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = _jwt_verify(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = get_user_by_id(payload["sub"])
    if not user or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or disabled.",
        )
    return user


def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[dict]:
    """Like get_current_user but returns None instead of raising for unauthenticated requests."""
    if not token:
        return None
    try:
        payload = _jwt_verify(token)
        return get_user_by_id(payload["sub"])
    except Exception:
        return None
