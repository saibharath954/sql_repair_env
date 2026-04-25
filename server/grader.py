"""
Deterministic grader for the SQL Repair environment.

No LLM is involved. Scoring is 100% programmatic and reproducible:
given identical database state, the score is always identical.

Each task has up to 5 sub-goals plus a faults_diagnosed meta-goal.
Partial credit accumulates as the agent progresses.

Design notes for single-shot RL training:
- The grader works for both iterative agents (multi-step inspect-then-fix)
  and single-shot agents (one submit_query). For single-shot agents we
  auto-grant `schema_inspected` (medium) when the result columns look
  correct, and we auto-grant `duplicates_detected`/`type_cast_present`
  (hard) when the SQL itself shows the agent recognised those issues.
- Score is in [0.0, 1.0]. We do NOT clamp empty results to 0.001 anymore
  because that destroys GRPO advantage signal across rollouts.
"""

import re
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


def _to_number(v):
    """Best-effort numeric coercion. Returns float when possible, else original."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    return v


def _normalise_value(v):
    """Round floats to 2dp; coerce numeric strings to floats; passthrough else."""
    n = _to_number(v)
    if isinstance(n, float):
        return round(n, 2)
    return n


def _rows_equal(actual: list, expected: list, columns: list) -> bool:
    """Compare two result sets regardless of internal dict ordering, with
    numeric-string tolerance (TYPE_DRIFT injects TEXT-stored numerics)."""
    if len(actual) != len(expected):
        return False

    # FAST-FAIL: every expected column must exist on the agent's rows.
    if actual and not all(c in actual[0] for c in columns):
        return False

    def normalise(row):
        return tuple(_normalise_value(row[c]) for c in columns)

    try:
        return sorted(normalise(r) for r in actual) == sorted(normalise(r) for r in expected)
    except Exception:
        return False


def _cols_present(actual: list, expected_columns: list) -> bool:
    if not actual:
        return False
    return all(c in actual[0] for c in expected_columns)


# Regex helpers used to grant single-shot-only sub-goals from the SQL itself.
_RE_DISTINCT      = re.compile(r"\bDISTINCT\b|\bGROUP\s+BY\b", re.IGNORECASE)
_RE_CAST          = re.compile(r"\bCAST\s*\(|\bAS\s+REAL\b|\*\s*1\.0|\+\s*0\.0", re.IGNORECASE)
_RE_INNER_JOIN    = re.compile(r"\bINNER\s+JOIN\b|\bJOIN\b\s+\w+\s+\w+\s+ON\s+", re.IGNORECASE)
_RE_REF_PRODUCTS  = re.compile(r"\bproducts\b", re.IGNORECASE)
_RE_REF_ORDERS    = re.compile(r"\borders\b", re.IGNORECASE)


def compute_score(
    conn: Optional[sqlite3.Connection],
    task_id: str,
    last_result: Optional[list] = None,
    achieved_flags: Optional[Dict[str, bool]] = None,
    submitted_sql: Optional[str] = None,
) -> Tuple[float, Dict[str, bool]]:
    """
    Compute the current grader score (0.0–1.0).

    Args:
        conn: live sqlite connection (used by the env to set flags out-of-band).
        task_id: 'easy' | 'medium' | 'hard'.
        last_result: most recent query result (list of dicts) from submit_query.
        achieved_flags: mutable dict of subgoal flags carried across the episode.
        submitted_sql: the raw SQL string the agent just submitted, if any.
            Used to grant single-shot-only sub-goals (CAST/DISTINCT detection,
            schema_inspected via referenced tables, etc.).

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
    sql = (submitted_sql or "").strip()

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
        if result and not any("error" in str(r).lower() for r in result):
            achieved_flags["query_executes_without_error"] = True
        if result and _cols_present(result, expected_cols):
            achieved_flags["correct_columns_returned"] = True
        if len(result) == len(expected):
            achieved_flags["correct_row_count"] = True
        if _rows_equal(result, expected, expected_cols):
            achieved_flags["correct_values"] = True

        # Single-shot ergonomics: if the SQL evidently references both
        # tables AND returns correct columns, the agent has clearly inspected
        # / understood the schema. Auto-grant the flag so a perfect single-
        # shot submit can reach 1.0.
        if not achieved_flags.get("schema_inspected"):
            sql_refs_both = bool(
                _RE_REF_ORDERS.search(sql) and _RE_REF_PRODUCTS.search(sql)
            )
            if sql_refs_both and (
                achieved_flags.get("correct_columns_returned")
                or achieved_flags.get("correct_row_count")
            ):
                achieved_flags["schema_inspected"] = True

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

        # Single-shot ergonomics: detect the deduplication and type-cast
        # signals directly in the submitted SQL — these were originally only
        # set by the env when the agent ran inspect_data / used CAST.
        if sql:
            if _RE_CAST.search(sql):
                achieved_flags["type_cast_present"] = True
            if _RE_DISTINCT.search(sql):
                achieved_flags["duplicates_detected"] = True

    # ── faults_diagnosed meta-goal ────────────────────────────────────────
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

    # Strict [0.0, 1.0]. NO floor: a completely failed query MUST score 0.0
    # so GRPO can compute meaningful advantages across rollouts.
    safe_score = max(0.0, min(raw_score, 1.0))

    return round(safe_score, 4), achieved_flags


def compute_potential(achieved_flags: Dict[str, bool], task_id: str) -> float:
    """Return the potential Φ(s) used for dense reward shaping."""
    raw_score = sum(
        (weight for label, weight in SUBGOALS[task_id] if achieved_flags.get(label, False)),
        0.0,
    )
    return round(max(0.0, min(raw_score, 1.0)), 4)
