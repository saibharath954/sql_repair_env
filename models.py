"""
SQL Repair Environment — type-safe Pydantic models.

These classes define the contract between agents and the environment.
Agents inspect the action schema via GET /tasks and submit structured actions via POST /step.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation, State


@dataclass
class SQLRepairAction(Action):
    """
    Action for the SQL Repair environment.

    Choose one action_type per step:

    - submit_query   : Execute a SQL query. Provide sql_query.
    - query_schema   : Inspect the DDL schema of a table. Provide target_table.
    - inspect_data   : Preview the first N rows of a table. Provide target_table.
    - list_tables    : List all tables in the current database. No extra fields needed.
    - run_test       : Run the hidden test suite against your last submitted query.
    """

    action_type: str = ""
    """One of: submit_query | query_schema | inspect_data | list_tables | run_test"""

    sql_query: Optional[str] = None
    """SQL string to execute. Required for submit_query."""

    target_table: Optional[str] = None
    """Table name. Required for query_schema and inspect_data."""


@dataclass
class SQLRepairObservation(Observation):
    """
    Observation returned after every action.

    Fields:
    - query_result   : List of row dicts from the last SELECT (empty for DDL/errors).
    - error_message  : Non-empty string if the last action raised an error.
    - schema_info    : DDL string after a query_schema action.
    - rows_affected  : Count of rows changed by INSERT/UPDATE/DELETE.
    - partial_score  : Current grader score for this episode (0.0–1.0). Rises as you fix things.
    - hint           : Optional hint nudge when score hasn't progressed in several steps.
    - step_count     : Steps taken so far in this episode.
    - tables         : List of table names, populated after list_tables action.
    - task_id        : Which task is active (easy | medium | hard).
    - task_description: Human-readable description of the goal.
    - broken_query   : The original broken SQL provided at episode start.
    """

    query_result: List[Dict[str, Any]] = field(default_factory=list)
    error_message: str = ""
    schema_info: str = ""
    rows_affected: int = 0
    partial_score: float = 0.0
    hint: str = ""
    step_count: int = 0
    tables: List[str] = field(default_factory=list)
    task_id: str = ""
    task_description: str = ""
    broken_query: str = ""


@dataclass
class SQLRepairState(State):
    """Episode metadata."""
    task_id: str = ""
    custom_field: int = 0