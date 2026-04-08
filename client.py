"""
SQLRepairEnv — typed OpenEnv client.

Install from the HF Space URL:
    pip install git+https://huggingface.co/spaces/YOUR_USERNAME/sql-repair-env

Usage (async):
    from sql_repair_env import SQLRepairAction, SQLRepairEnv
    async with SQLRepairEnv(base_url="https://YOUR_SPACE.hf.space") as env:
        result = await env.reset(task_id="easy")
        result = await env.step(SQLRepairAction(
            action_type="submit_query",
            sql_query="SELECT name, salary FROM employees WHERE dept='Engineering' ORDER BY salary DESC"
        ))
        print(result.observation.partial_score)

Usage (sync):
    with SQLRepairEnv(base_url="...").sync() as env:
        result = env.reset(task_id="medium")
        result = env.step(SQLRepairAction(action_type="list_tables"))
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SQLRepairAction, SQLRepairObservation, SQLRepairState


class SQLRepairEnv(EnvClient[SQLRepairAction, SQLRepairObservation, SQLRepairState]):
    """Typed client for the SQL Repair environment."""

    def _step_payload(self, action: SQLRepairAction) -> dict:
        return {
            "action_type": action.action_type,
            "sql_query":   action.sql_query,
            "target_table": action.target_table,
        }

    def _parse_result(self, payload: dict) -> StepResult[SQLRepairObservation]:
        obs_data = payload.get("observation", payload)
        obs = SQLRepairObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            query_result=obs_data.get("query_result", []),
            error_message=obs_data.get("error_message", ""),
            schema_info=obs_data.get("schema_info", ""),
            rows_affected=obs_data.get("rows_affected", 0),
            partial_score=obs_data.get("partial_score", 0.0),
            hint=obs_data.get("hint", ""),
            step_count=obs_data.get("step_count", 0),
            tables=obs_data.get("tables", []),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            broken_query=obs_data.get("broken_query", ""),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SQLRepairState:
        return SQLRepairState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
        )