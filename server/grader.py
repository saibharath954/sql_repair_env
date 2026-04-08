"""
Deterministic grader for the SQL Repair environment.

No LLM is involved. Scoring is 100% programmatic and reproducible:
given identical database state, the score is always identical.

Each task has up to 5 sub-goals. Partial credit accumulates as the agent
progresses. The grader also tracks WHICH sub-goals have been achieved so
the reward-shaping logic can compute delta rewards correctly.
"""

import sqlite3
from typing import Dict, Optional, Tuple


# ─── sub-goal weights per task ───────────────────────────────────────────────
# Each entry is (label, weight). Weights sum to 1.0.
SUBGOALS = {
    "easy": [
        ("query_executes_without_error", 0.25),
        ("correct_columns_returned",     0.25),
        ("correct_row_count",            0.25),
        ("correct_values",               0.25),
    ],
    "medium": [
        ("schema_inspected",             0.15),
        ("query_executes_without_error", 0.20),
        ("correct_columns_returned",     0.15),
        ("correct_row_count",            0.20),
        ("correct_values",               0.30),
    ],
    "hard": [
        ("duplicates_detected",          0.15),
        ("type_cast_present",            0.15),
        ("invalid_fk_excluded",          0.10),
        ("correct_row_count",            0.20),
        ("correct_values",               0.40),
    ],
}


def _rows_equal(actual: list, expected: list, columns: list) -> bool:
    """Compare two result sets regardless of internal dict ordering."""
    if len(actual) != len(expected):
        return False
    def normalise(row):
        return tuple(round(float(row[c]), 2) if isinstance(row[c], float) else row[c]
                     for c in columns)
    return sorted(normalise(r) for r in actual) == sorted(normalise(r) for r in expected)


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

    Args:
        conn            : Active SQLite connection for the episode.
        task_id         : "easy" | "medium" | "hard"
        last_result     : Rows returned by the agent's most recent SELECT.
        achieved_flags  : Dict tracking which sub-goals have been permanently achieved.
                          Once True, a flag is never reset to False (monotonic progress).

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
        # duplicates_detected: agent ran a query that discovered duplicate txn_ids
        # (set externally by environment when it sees a DISTINCT / GROUP BY on txn_id)

        # type_cast_present: inspect last sql for CAST or amount*1.0 or REAL
        # (set externally by environment when it sees the cast in sql_query)

        # invalid_fk_excluded: check no customer_id=99 in result
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

    # ── compute weighted total ─────────────────────────────────────────────
    score = sum(
        (weight for label, weight in SUBGOALS[task_id] if achieved_flags.get(label, False)),
        0.0  # Force it to be a float!
    )
    return round(min(score, 1.0), 4), achieved_flags


def compute_potential(achieved_flags: Dict[str, bool], task_id: str) -> float:
    """Return the potential Φ(s) used for dense reward shaping."""
    score = sum(
        (weight for label, weight in SUBGOALS[task_id] if achieved_flags.get(label, False)),
        0.0  # Force it to be a float!
    )
    return round(min(score, 1.0), 4)