"""Persistent, input-sensitive terminal states for observed-memory work."""

from __future__ import annotations

import json
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"success", "exhausted", "unrunnable"}
TRANSIENT_ERROR_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "connection refused",
    "connection reset",
    "remote end closed",
    "timed out",
    "timeout",
    "temporary failure",
    "rate limit",
    "service unavailable",
    "ssl:",
    "unexpected_eof",
    "transport_error",
)
NON_RETRYABLE_ERROR_MARKERS = (
    "invalid_prompt",
    "http 400",
    "400 bad request",
)


def fingerprint(parts: dict[str, Any], files: list[Path]) -> str:
    digest = sha256()
    digest.update(
        json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    for path in files:
        digest.update(b"\0")
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


def load_status(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def reusable_terminal_status(
    path: Path,
    *,
    expected_fingerprint: str,
    min_budget: dict[str, int],
) -> dict | None:
    data = load_status(path)
    if not data or data.get("status") not in TERMINAL_STATUSES:
        return None
    if data.get("fingerprint") != expected_fingerprint:
        return None
    budget = data.get("budget")
    if not isinstance(budget, dict):
        return None
    if data["status"] != "success":
        for key, required in min_budget.items():
            value = budget.get(key)
            if not isinstance(value, int) or value < required:
                return None
    return data


def write_status(
    path: Path,
    *,
    status: str,
    fingerprint_value: str,
    budget: dict[str, int],
    model: str,
    reason: str = "",
    details: dict[str, Any] | None = None,
) -> dict:
    payload = {
        "status": status,
        "fingerprint": fingerprint_value,
        "budget": budget,
        "model": model,
        "reason": reason,
        "updated_at": datetime.now().isoformat(),
    }
    if details:
        payload["details"] = details
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def is_transient_error(error: str) -> bool:
    lowered = error.lower()
    return any(marker in lowered for marker in TRANSIENT_ERROR_MARKERS)


def is_non_retryable_error(error: str) -> bool:
    lowered = error.lower()
    return any(marker in lowered for marker in NON_RETRYABLE_ERROR_MARKERS)


def normalized_error_signature(error: str) -> str:
    """Collapse volatile paths and identifiers in repeated error messages."""
    value = error.lower().strip()
    value = re.sub(r"/tmp/[^\s:'\"]+", "<tmp>", value)
    value = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "<uuid>",
        value,
    )
    value = re.sub(r"line \d+", "line <n>", value)
    value = re.sub(r"\b0x[0-9a-f]+\b", "<address>", value)
    value = re.sub(r"\s+", " ", value)
    return value[:500]
