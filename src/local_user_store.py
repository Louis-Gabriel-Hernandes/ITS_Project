from __future__ import annotations

import base64
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .student_model import JAVA_TOPICS, new_student_model
from .utils import ensure_dir, project_root

DB_PATH = project_root() / "data" / "local" / "java_its_users.db"
PBKDF2_ITERATIONS = 260_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection() -> sqlite3.Connection:
    ensure_dir(DB_PATH.parent)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_progress (
                user_id INTEGER PRIMARY KEY,
                progress_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    if salt is None:
        salt = os.urandom(16)
    digest = base64.b64encode(
        __import__("hashlib").pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
        )
    ).decode("ascii")
    salt_text = base64.b64encode(salt).decode("ascii")
    return salt_text, digest


def _verify_password(password: str, salt_text: str, expected_hash: str) -> bool:
    salt = base64.b64decode(salt_text.encode("ascii"))
    _, actual = _hash_password(password, salt)
    return __import__("hmac").compare_digest(actual, expected_hash)


def user_public(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "email": row["email"] or "",
        "display_name": row["display_name"],
    }


def create_user(username: str, password: str, display_name: str = "", email: str = "") -> Dict[str, Any]:
    init_db()
    username = username.strip()
    email = email.strip() or None
    display_name = display_name.strip() or username

    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    salt, digest = _hash_password(password)
    now = utc_now()
    try:
        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO users (username, email, display_name, password_salt, password_hash, created_at, last_login_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (username, email, display_name, salt, digest, now, now),
            )
            user_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO user_progress (user_id, progress_json, updated_at) VALUES (?, ?, ?)",
                (user_id, json.dumps(default_progress()), now),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return user_public(row)
    except sqlite3.IntegrityError as exc:
        message = str(exc).lower()
        if "email" in message:
            raise ValueError("That email is already registered.") from exc
        raise ValueError("That username is already taken.") from exc


def authenticate_user(username_or_email: str, password: str) -> Optional[Dict[str, Any]]:
    init_db()
    lookup = username_or_email.strip()
    if not lookup or not password:
        return None
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM users
            WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE
            """,
            (lookup, lookup),
        ).fetchone()
        if row is None:
            return None
        if not _verify_password(password, row["password_salt"], row["password_hash"]):
            return None
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (utc_now(), row["id"]))
        return user_public(row)


def ensure_student_defaults(student: Dict[str, Any] | None) -> Dict[str, Any]:
    base = new_student_model()
    if isinstance(student, dict):
        base.update(student)

    base.setdefault("mastery", {})
    base.setdefault("attempts_by_topic", {})
    base.setdefault("correct_by_topic", {})
    base.setdefault("hint_usage_by_topic", {})
    for topic in JAVA_TOPICS:
        base["mastery"].setdefault(topic, 0.20)
        base["attempts_by_topic"].setdefault(topic, 0)
        base["correct_by_topic"].setdefault(topic, 0)
        base["hint_usage_by_topic"].setdefault(topic, 0)

    # This app starts new users at the easiest level and uses sessions, not an old diagnostic phase.
    base["phase"] = "adaptive"
    base["diagnostic_remaining"] = 0
    base["is_new_student"] = bool(base.get("attempts", 0) == 0)
    base["current_difficulty"] = max(1, min(5, int(base.get("current_difficulty", 1) or 1)))
    base.setdefault("answered_question_ids", [])
    base.setdefault("recent_history", [])
    base.setdefault("misconceptions", {})
    base.setdefault("current_streak", 0)
    base.setdefault("confidence", sum(base["mastery"].values()) / max(1, len(base["mastery"])))
    base.setdefault("recommended_next_topic", "variables")
    return base


def default_progress() -> Dict[str, Any]:
    student = ensure_student_defaults(new_student_model())
    return {
        "student": student,
        "answered_question_ids": [],
        "topic_filter": "__mixed__",
        "session_number": 1,
        "session_difficulty": 1,
        "session_answered_ids": [],
        "session_results": [],
        "session_adjusted": False,
        "current_question": None,
        "hint_level": 0,
        "feedback": None,
        "show_explanation": False,
        "answer_nonce": 0,
        "best_streak": 0,
    }


def load_progress(user_id: int) -> Dict[str, Any]:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT progress_json FROM user_progress WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            progress = default_progress()
            conn.execute(
                "INSERT INTO user_progress (user_id, progress_json, updated_at) VALUES (?, ?, ?)",
                (user_id, json.dumps(progress), utc_now()),
            )
            return progress
        try:
            progress = json.loads(row["progress_json"])
        except json.JSONDecodeError:
            progress = default_progress()

    default = default_progress()
    default.update(progress if isinstance(progress, dict) else {})
    default["student"] = ensure_student_defaults(default.get("student"))
    answered = set(default.get("answered_question_ids", [])) | set(default["student"].get("answered_question_ids", []))
    default["answered_question_ids"] = list(answered)
    default["student"]["answered_question_ids"] = list(answered)
    default["session_difficulty"] = max(1, min(5, int(default.get("session_difficulty", default["student"].get("current_difficulty", 1)) or 1)))
    default["session_results"] = list(default.get("session_results", []))
    default["session_answered_ids"] = list(default.get("session_answered_ids", []))
    default["topic_filter"] = default.get("topic_filter") or "__mixed__"
    return default


def save_progress(user_id: int, progress: Dict[str, Any]) -> None:
    init_db()
    safe_progress = dict(progress)
    safe_progress["student"] = ensure_student_defaults(safe_progress.get("student"))
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_progress (user_id, progress_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                progress_json = excluded.progress_json,
                updated_at = excluded.updated_at
            """,
            (user_id, json.dumps(safe_progress), utc_now()),
        )


def reset_progress(user_id: int) -> Dict[str, Any]:
    progress = default_progress()
    save_progress(user_id, progress)
    return progress
