"""
SQLite persistence layer for DeepDetect analysis jobs.

Replaces the in-memory `jobs` dict in api.py with a durable store that
survives API restarts.  Jobs still run in-process; only the metadata and
results are persisted.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

DB_PATH: str = os.getenv(
    "DEEPDETECT_DB",
    str(Path(__file__).resolve().parent.parent / "deepdetect.db"),
)

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist. Call once at startup."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS analysis_jobs (
                id              TEXT PRIMARY KEY,
                user_id         TEXT,
                status          TEXT NOT NULL DEFAULT 'queued',
                created_at      TEXT NOT NULL,
                started_at      TEXT,
                completed_at    TEXT,
                filename        TEXT NOT NULL,
                source          TEXT NOT NULL,
                video_path      TEXT NOT NULL,
                num_frames      INTEGER NOT NULL DEFAULT 32,
                metadata_json   TEXT,
                result_json     TEXT,
                error           TEXT,
                batch_id        TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_user ON analysis_jobs(user_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_batch ON analysis_jobs(batch_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON analysis_jobs(status);
        """)


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def insert_job(
    job_id: str,
    filename: str,
    source: str,
    video_path: str,
    num_frames: int,
    metadata: dict,
    user_id: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO analysis_jobs
               (id, user_id, status, created_at, filename, source, video_path,
                num_frames, metadata_json, batch_id)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                job_id, user_id, "queued", now, filename, source, video_path,
                num_frames, json.dumps(metadata), batch_id,
            ),
        )
        conn.commit()


def set_job_running(job_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE analysis_jobs SET status='running', started_at=? WHERE id=?",
            (now, job_id),
        )
        conn.commit()


def set_job_completed(job_id: str, result: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE analysis_jobs SET status='completed', completed_at=?, result_json=? WHERE id=?",
            (now, json.dumps(result), job_id),
        )
        conn.commit()


def set_job_failed(job_id: str, error: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE analysis_jobs SET status='failed', completed_at=?, error=? WHERE id=?",
            (now, error, job_id),
        )
        conn.commit()


def fail_stale_jobs(error: str = "Analysis worker restarted before completing.") -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE analysis_jobs SET status='failed', completed_at=?, error=? "
            "WHERE status IN ('queued', 'running')",
            (now, error),
        )
        conn.commit()


def get_job(job_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_jobs WHERE id=?", (job_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_jobs_for_user(user_id: str, limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_jobs WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_jobs_in_batch(batch_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_jobs WHERE batch_id=? ORDER BY created_at ASC",
            (batch_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_job(job_id: str) -> Optional[str]:
    """Delete job record; return video_path for file cleanup."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT video_path FROM analysis_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM analysis_jobs WHERE id=?", (job_id,))
        conn.commit()
    return row["video_path"]


def get_all_jobs(limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def create_batch_id() -> str:
    import uuid
    return uuid.uuid4().hex


def get_batch_summary(batch_id: str) -> dict:
    jobs = get_jobs_in_batch(batch_id)
    statuses = [j["status"] for j in jobs]
    return {
        "batch_id": batch_id,
        "total": len(jobs),
        "queued": statuses.count("queued"),
        "running": statuses.count("running"),
        "completed": statuses.count("completed"),
        "failed": statuses.count("failed"),
        "jobs": [_public_job(j) for j in jobs],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("metadata_json", "result_json"):
        raw = d.pop(key, None)
        field = key.replace("_json", "")
        d[field] = json.loads(raw) if raw else None
    return d


def _public_job(job: dict) -> dict:
    """Strip internal video_path; add video_url for API consumers."""
    out = {k: v for k, v in job.items() if k != "video_path"}
    result = out.get("result")
    if isinstance(result, dict):
        suspicious = result.get("suspicious")
        if isinstance(suspicious, dict):
            if "suspicious_frames" not in suspicious:
                suspicious["suspicious_frames"] = suspicious.get("frames", [])
            suspicious.setdefault("suspicious_count", len(suspicious["suspicious_frames"]))
            suspicious.setdefault("total_frames", result.get("frames_analyzed", 0))
            suspicious.setdefault("fps", 0.0)
            suspicious.setdefault("detector_name", result.get("detector_name", "Unknown"))
            suspicious.setdefault("frame_scores", result.get("frame_scores", []))
            if not suspicious["suspicious_frames"] and suspicious["frame_scores"]:
                suspicious["suspicious_frames"] = [
                    {
                        "frame_index": index,
                        "timestamp_sec": index / (suspicious["fps"] or 25.0),
                        "fake_score": score,
                        "neighbour_mse": 0.0,
                        "neighbour_delta": 0.0,
                        "reason": "Sampled frame score is above the 50% screening threshold.",
                        "face_data_url": (
                            result.get("face_frames", [])[index]
                            if index < len(result.get("face_frames", []))
                            else ""
                        ),
                    }
                    for index, score in enumerate(suspicious["frame_scores"])
                    if score >= 0.5
                ]
                suspicious["suspicious_count"] = len(suspicious["suspicious_frames"])
            suspicious.setdefault("evidence_frames", suspicious["suspicious_frames"])
        out["result"] = result
    out["video_url"] = f"/api/analyses/{job['id']}/video"
    return out
