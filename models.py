"""
SQL Repair Environment — type-safe Pydantic models.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation, State

class SQLRepairAction(Action):
    """Action for the SQL Repair environment."""
    action_type: str = ""
    sql_query: Optional[str] = None
    target_table: Optional[str] = None


class SQLRepairObservation(Observation):
    """Observation returned after every action."""
    query_result: List[Dict[str, Any]] = []
    error_message: str = ""
    schema_info: str = ""
    rows_affected: int = 0
    partial_score: float = 0.0
    hint: str = ""
    step_count: int = 0
    tables: List[str] = []
    task_id: str = ""
    task_description: str = ""
    broken_query: str = ""
    
    # --- MISSING FIELDS ADDED BELOW ---
    done: bool = False
    reward: float = 0.0


class SQLRepairState(State):
    """Episode metadata."""
    episode_id: str
    task_id: str = ""
    step_count: int = 0