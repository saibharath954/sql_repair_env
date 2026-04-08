"""
SQL Repair Environment — core server-side logic.

Implements the three OpenEnv methods:
  reset()  → SQLRepairObservation
  step()   → SQLRepairObservation   (reward & done embedded per RFC 002)
  state    → SQLRepairState         (@property)

Reward design  (potential-based, dense):
  R(t) = Φ(s_t) − Φ(s_{t-1}) − STEP_PENALTY
  Terminal bonus +1.0 on perfect score.
  Penalty −0.5 for destructive SQL (DROP / TRUNCATE / DELETE without WHERE).
  Penalty −0.3 for syntax errors that loop.
"""

import re
import sqlite3
import sys
import os
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Resolve imports whether run as package or directly (uvicorn server.app:app)
# Add repo root to sys.path so 'models' is always importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from models import SQLRepairAction, SQLRepairObservation, SQLRepairState

try:
    from server.tasks import TASKS
    from server.grader import compute_potential, compute_score
except ModuleNotFoundError:
    from tasks import TASKS
    from grader import compute_potential, compute_score

MAX_STEPS = 20
STEP_PENALTY = 0.02
DESTRUCTIVE_RE = re.compile(
    r"\b(DROP\s+TABLE|TRUNCATE|DELETE\s+FROM\s+\w+\s*(?!WHERE))\b",
    re.IGNORECASE,
)
INSPECT_PREVIEW_ROWS = 8


class SQLRepairEnvironment(Environment):
    """Sandboxed SQLite-based SQL repair environment."""

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self._task_id: str = "easy"
        self._state: SQLRepairState = SQLRepairState(
            episode_id=str(uuid.uuid4()), task_id="easy", step_count=0
        )
        self._achieved_flags: Dict[str, bool] = {}
        self._prev_potential: float = 0.0
        self._hint_index: int = 0
        self._last_error_count: int = 0

    # ─── OpenEnv interface ────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> SQLRepairObservation:
        """Start a fresh episode. Tears down old DB, injects task-specific faults."""
        if task_id not in TASKS:
            task_id = "easy"

        # Close previous in-memory DB
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass

        # Fresh in-memory SQLite — perfectly isolated per episode
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row

        task = TASKS[task_id]
        self._conn.executescript(task["setup_sql"])
        self._conn.commit()

        self._task_id = task_id
        self._state = SQLRepairState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
        )
        self._achieved_flags = {k: False for k, _ in self._get_subgoals()}
        self._prev_potential = 0.0
        self._hint_index = 0
        self._last_error_count = 0

        return SQLRepairObservation(
            done=False,
            reward=0.0,
            query_result=[],
            error_message="",
            schema_info="",
            rows_affected=0,
            partial_score=0.0,
            hint=f"TASK ({task_id.upper()}): {task['description']}\n\nBROKEN QUERY:\n{task['broken_query']}",
            step_count=0,
            tables=[],
            task_id=task_id,
            task_description=task["description"],
            broken_query=task["broken_query"],
        )

    def step(self, action: SQLRepairAction) -> SQLRepairObservation:
        """Execute one agent action and return the resulting observation with reward."""
        self._state.step_count += 1
        step = self._state.step_count

        obs = self._dispatch(action)

        # ── grader score after this action ────────────────────────────────
        score, self._achieved_flags = compute_score(
            conn=self._conn,
            task_id=self._task_id,
            last_result=obs.query_result,
            achieved_flags=self._achieved_flags,
        )
        current_potential = compute_potential(self._achieved_flags, self._task_id)

        # ── dense reward ──────────────────────────────────────────────────
        reward = current_potential - self._prev_potential - STEP_PENALTY

        # Destructive SQL penalty
        if action.sql_query and DESTRUCTIVE_RE.search(action.sql_query):
            reward -= 0.5

        # Error loop penalty (discourage repeating broken queries)
        if obs.error_message:
            self._last_error_count += 1
            if self._last_error_count >= 3:
                reward -= 0.15
        else:
            self._last_error_count = 0

        self._prev_potential = current_potential

        # ── terminal conditions ───────────────────────────────────────────
        done = False
        if score >= 1.0:
            reward += 1.0  # full-solve bonus
            done = True
        elif step >= MAX_STEPS:
            done = True

        # ── progressive hints ─────────────────────────────────────────────
        hint = obs.hint
        task_hints = TASKS[self._task_id]["hints"]
        if not hint and current_potential == self._prev_potential:
            if step == 5 and self._hint_index == 0:
                hint = task_hints[0]; self._hint_index = 1
            elif step == 10 and self._hint_index <= 1:
                hint = task_hints[1]; self._hint_index = 2
            elif step == 15 and self._hint_index <= 2:
                hint = task_hints[2]; self._hint_index = 3

        obs.partial_score = score
        obs.step_count = step
        obs.done = done
        obs.reward = round(reward, 4)
        obs.hint = hint
        obs.task_id = self._task_id
        obs.task_description = TASKS[self._task_id]["description"]

        return obs

    @property
    def state(self) -> SQLRepairState:
        return self._state

    # ─── action dispatcher ────────────────────────────────────────────────────

    def _dispatch(self, action: SQLRepairAction) -> SQLRepairObservation:
        atype = (action.action_type or "").strip().lower()

        if atype == "submit_query":
            return self._submit_query(action.sql_query or "")

        elif atype == "query_schema":
            return self._query_schema(action.target_table or "")

        elif atype == "inspect_data":
            return self._inspect_data(action.target_table or "")

        elif atype == "list_tables":
            return self._list_tables()

        elif atype == "run_test":
            return self._run_test()

        else:
            return SQLRepairObservation(
                error_message=(
                    f"Unknown action_type '{action.action_type}'. "
                    "Valid types: submit_query | query_schema | inspect_data | list_tables | run_test"
                )
            )

    # ─── action implementations ───────────────────────────────────────────────

    def _submit_query(self, sql: str) -> SQLRepairObservation:
        sql = sql.strip()
        if not sql:
            return SQLRepairObservation(error_message="sql_query is required for submit_query.")
        try:
            cursor = self._conn.execute(sql)
            self._conn.commit()
            if cursor.description:
                cols = [d[0] for d in cursor.description]
                rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
                return SQLRepairObservation(query_result=rows, rows_affected=len(rows))
            else:
                return SQLRepairObservation(rows_affected=cursor.rowcount)
        except Exception as e:
            return SQLRepairObservation(error_message=f"SQL Error: {e}")

    def _query_schema(self, table: str) -> SQLRepairObservation:
        # Mark schema-inspected flag for medium task
        if self._task_id == "medium" and table:
            self._achieved_flags["schema_inspected"] = True
        try:
            rows = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchall()
            if not rows:
                return SQLRepairObservation(
                    error_message=f"Table '{table}' not found. Use list_tables to see available tables."
                )
            return SQLRepairObservation(schema_info=rows[0][0])
        except Exception as e:
            return SQLRepairObservation(error_message=str(e))

    def _inspect_data(self, table: str) -> SQLRepairObservation:
        # Hard task: if agent inspects transactions and data has duplicate txn_ids, flag it
        try:
            cursor = self._conn.execute(f'SELECT * FROM "{table}" LIMIT {INSPECT_PREVIEW_ROWS}')
            cols = [d[0] for d in cursor.description]
            rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
            hint = f"Showing up to {INSPECT_PREVIEW_ROWS} rows from '{table}'."

            if self._task_id == "hard" and table in ("transactions",):
                # Check if duplicates visible
                dup_check = self._conn.execute(
                    "SELECT txn_id, COUNT(*) as cnt FROM transactions GROUP BY txn_id HAVING cnt > 1"
                ).fetchall()
                if dup_check:
                    self._achieved_flags["duplicates_detected"] = True
                    hint += f" WARNING: {len(dup_check)} duplicate txn_id(s) detected."

            return SQLRepairObservation(query_result=rows, hint=hint)
        except Exception as e:
            return SQLRepairObservation(error_message=f"inspect_data error: {e}")

    def _list_tables(self) -> SQLRepairObservation:
        try:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            tables = [r[0] for r in rows]
            return SQLRepairObservation(tables=tables, hint=f"Tables: {', '.join(tables)}")
        except Exception as e:
            return SQLRepairObservation(error_message=str(e))

    def _run_test(self) -> SQLRepairObservation:
        """Run the hidden test against the expected result and report per-subgoal breakdown."""
        task = TASKS[self._task_id]
        correct_sql = task["action_schema"]["sql_query"]
        try:
            cursor = self._conn.execute(correct_sql)
            cols = [d[0] for d in cursor.description]
            rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
            score, flags = compute_score(
                conn=self._conn,
                task_id=self._task_id,
                last_result=rows,
                achieved_flags=dict(self._achieved_flags),
            )
            breakdown = {k: "✓" if v else "✗" for k, v in flags.items()}
            return SQLRepairObservation(
                query_result=rows,
                hint=f"Test result: score={score:.2f} | Sub-goals: {breakdown}",
                partial_score=score,
            )
        except Exception as e:
            return SQLRepairObservation(error_message=f"run_test error: {e}")

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _get_subgoals(self):
        try:
            from server.grader import SUBGOALS
        except ModuleNotFoundError:
            from grader import SUBGOALS
        return SUBGOALS.get(self._task_id, [])

    def get_achieved_flags(self) -> Dict[str, bool]:
        return dict(self._achieved_flags)

    def get_current_score(self) -> float:
        score, _ = compute_score(
            conn=self._conn,
            task_id=self._task_id,
            last_result=[],
            achieved_flags=dict(self._achieved_flags),
        )
        return score

    def mark_type_cast_present(self):
        """Called by app.py when it detects CAST in submitted SQL (hard task)."""
        if self._task_id == "hard":
            self._achieved_flags["type_cast_present"] = True