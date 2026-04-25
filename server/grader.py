"""
Deterministic grader for the SQL Repair environment.

No LLM is involved. Scoring is 100% programmatic and reproducible:
given identical database state, the score is always identical.

Each task has up to 5 sub-goals plus a faults_diagnosed meta-goal.
Partial credit accumulates as the agent progresses.
"""

import sqlite3
from typing import Dict, Optional, Tuple


# ─── sub-goal weights per task ───────────────────────────────────────────────
# Each entry is (label, weight). Weights sum to 1.0.
# faults_diagnosed (0.10) is achieved when the other subgoals hit >= 0.80.
SUBGOALS = {
    "easy": [
        ("query_executes_without_error", 0.225),
        ("correct_columns_returned",     0.225),
        ("correct_row_count",            0.225),
        ("correct_values",               0.225),
        ("faults_diagnosed",             0.10),
    ],
    "medium": [
        ("schema_inspected",             0.135),
        ("query_executes_without_error", 0.18),
        ("correct_columns_returned",     0.135),
        ("correct_row_count",            0.18),
        ("correct_values",               0.27),
        ("faults_diagnosed",             0.10),
    ],
    "hard": [
        ("duplicates_detected",          0.135),
        ("type_cast_present",            0.135),
        ("invalid_fk_excluded",          0.09),
        ("correct_row_count",            0.18),
        ("correct_values",               0.36),
        ("faults_diagnosed",             0.10),
    ],
}


def _rows_equal(actual: list, expected: list, columns: list) -> bool:
    """Compare two result sets regardless of internal dict ordering."""
    if len(actual) != len(expected):
        return False
        
    # 1. FAST-FAIL: Ensure all expected columns actually exist in the agent's results
    # This prevents the KeyError: 'total_spend' and KeyError: 'revenue'
    if actual and not all(c in actual[0] for c in columns):
        return False

    def normalise(row):
        return tuple(round(float(row[c]), 2) if isinstance(row[c], float) else row[c]
                     for c in columns)
                     
    # 2. SAFETY NET: Catch any rogue type-casting errors (e.g., if a weird string can't be float()'d)
    try:
        return sorted(normalise(r) for r in actual) == sorted(normalise(r) for r in expected)
    except Exception:
        return False


def _cols_present(actual: list, expected_columns: list) -> bool:
    if not actual:
        return False
    return all(c in actual[0] for c in expected_columns)


def compute_score(
    conn: Optional[sqlite3.Connection],
    task_id: str,
    last_result: Optional[list] = None,
    achieved_flags: Optional[Dict[str, bool]] = None,
) -> Tuple[float, Dict[str, bool]]:
    """
    Compute the current grader score (0.0–1.0).

    Returns:
        (score, updated_flags)
    """
    try:
        from server.tasks import TASKS
    except ModuleNotFoundError:
        from tasks import TASKS

    task = TASKS[task_id]
    expected = task["expected_rows"]
    expected_cols = task["expected_columns"]

    if achieved_flags is None:
        achieved_flags = {k: False for k, _ in SUBGOALS[task_id]}

    result = last_result or []

    # ── EASY ─────────────────────────────────────────────────────────────────
    if task_id == "easy":
        if result and not any("error" in str(r).lower() for r in result):
            achieved_flags["query_executes_without_error"] = True
        if result and _cols_present(result, expected_cols):
            achieved_flags["correct_columns_returned"] = True
        if len(result) == len(expected):
            achieved_flags["correct_row_count"] = True
        if _rows_equal(result, expected, expected_cols):
            achieved_flags["correct_values"] = True

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    elif task_id == "medium":
        # schema_inspected is set externally by the environment on query_schema action
        if result and not any("error" in str(r).lower() for r in result):
            achieved_flags["query_executes_without_error"] = True
        if result and _cols_present(result, expected_cols):
            achieved_flags["correct_columns_returned"] = True
        if len(result) == len(expected):
            achieved_flags["correct_row_count"] = True
        if _rows_equal(result, expected, expected_cols):
            achieved_flags["correct_values"] = True

    # ── HARD ──────────────────────────────────────────────────────────────────
    elif task_id == "hard":
        if result:
            has_orphan = any(
                str(r.get("customer_id", "")) == "99" for r in result
            )
            if not has_orphan and len(result) > 0:
                achieved_flags["invalid_fk_excluded"] = True

        if len(result) == len(expected):
            achieved_flags["correct_row_count"] = True
        if _rows_equal(result, expected, expected_cols):
            achieved_flags["correct_values"] = True

    # ── faults_diagnosed meta-goal ────────────────────────────────────────
    # Achieved when all other subgoals collectively score >= 0.80
    raw_score_without_fd = sum(
        weight for label, weight in SUBGOALS[task_id]
        if label != "faults_diagnosed" and achieved_flags.get(label, False)
    )
    if raw_score_without_fd >= 0.80:
        achieved_flags["faults_diagnosed"] = True

    # ── compute weighted total ─────────────────────────────────────────────
    raw_score = sum(
        (weight for label, weight in SUBGOALS[task_id] if achieved_flags.get(label, False)),
        0.0,
    )

    # Clamp score strictly between 0 and 1
    safe_score = max(0.001, min(raw_score, 0.999))

    return round(safe_score, 4), achieved_flags


def compute_potential(achieved_flags: Dict[str, bool], task_id: str) -> float:
    """Return the potential Φ(s) used for dense reward shaping."""
    raw_score = sum(
        (weight for label, weight in SUBGOALS[task_id] if achieved_flags.get(label, False)),
        0.0,
    )

    safe_score = max(0.001, min(raw_score, 0.999))

    return round(safe_score, 4)