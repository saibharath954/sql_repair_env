"""
FastAPI application for the SQL Repair Environment.

Routes:
  POST /reset            — start episode (task_id='auto' for curriculum-driven selection)
  POST /step             — execute action
  GET  /state            — episode metadata
  GET  /health           — liveness probe (HF Spaces ping target)
  GET  /tasks            — task list + action schemas  [required by hackathon]
  POST /grader           — grader score for current episode  [required by hackathon]
  POST /baseline         — run inference.py, return scores  [required by hackathon]
  GET  /faults           — reveal injected fault types (post-episode analysis)
  GET  /curriculum       — current difficulty level and progression stats
  POST /curriculum/reset — reset curriculum to Novice (testing only)
"""

import json
import os
import sys
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openenv.core.env_server import create_fastapi_app

# Ensure repo root is on sys.path so 'models' can be found
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from models import SQLRepairAction, SQLRepairObservation
from server.environment import SQLRepairEnvironment
from server.tasks import TASKS
from server.grader import SUBGOALS
from server.curriculum import CurriculumManager, LEVEL_NAMES

# ── create app ────────────────────────────────────────────────────────────────
env = SQLRepairEnvironment()
curriculum = CurriculumManager()
app: FastAPI = create_fastapi_app(lambda: env, SQLRepairAction, SQLRepairObservation)


# ── helper: serialise observation to dict ─────────────────────────────────────
def _obs_dict(obs: SQLRepairObservation) -> dict:
    return {
        "query_result":     obs.query_result,
        "error_message":    obs.error_message,
        "schema_info":      obs.schema_info,
        "rows_affected":    obs.rows_affected,
        "partial_score":    obs.partial_score,
        "hint":             obs.hint,
        "step_count":       obs.step_count,
        "tables":           obs.tables,
        "task_id":          obs.task_id,
        "task_description": obs.task_description,
        "broken_query":     obs.broken_query,
    }


# ─── /reset ───────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


@app.post("/reset")
async def reset_env(req: ResetRequest = ResetRequest()):
    """
    Reset the environment. Posting an empty body {} defaults to task_id='easy'.
    Pass task_id='auto' to let the curriculum manager choose task and difficulty.
    The pre-validation script posts {} — this MUST return HTTP 200.
    """
    suggestion = {}
    task_id = req.task_id

    if not task_id or task_id == "auto":
        suggestion = curriculum.suggest_task()
        task_id = suggestion["task_id"]
        env.set_difficulty_level(suggestion["difficulty_level"])

    obs = env.reset(task_id=task_id, seed=req.seed)
    return {
        "observation": _obs_dict(obs),
        "reward": 0.0,
        "done": False,
        "episode_id": env.state.episode_id,
        "injected_fault_count": env.state.injected_fault_count,
        "difficulty_level": env.state.difficulty_level,
        "level_name": suggestion.get("level_name", LEVEL_NAMES.get(env.state.difficulty_level, "Senior")),
    }


# ─── /step ────────────────────────────────────────────────────────────────────
@app.post("/step")
async def step_env(action: SQLRepairAction):
    """Execute one action and return the resulting observation with reward."""
    obs = env.step(action)

    # Record completed episodes for curriculum tracking
    promotion_info = {}
    if obs.done:
        promotion_info = curriculum.record_episode(
            task_id=env.state.task_id,
            score=obs.partial_score,
            steps_taken=env.state.step_count,
            injected_fault_count=env.state.injected_fault_count,
        )

    result = {
        "observation": _obs_dict(obs),
        "reward":      obs.reward,
        "done":        obs.done,
        "episode_id":  env.state.episode_id,
    }
    if promotion_info:
        result["curriculum"] = promotion_info
    return result


# ─── /state ───────────────────────────────────────────────────────────────────
@app.get("/state")
async def get_state():
    """Return current episode metadata."""
    s = env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "task_id": s.task_id,
        "injected_fault_count": s.injected_fault_count,
        "difficulty_level": s.difficulty_level,
    }


# ─── /health ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Liveness probe — HF Spaces automated ping / pre-validation target."""
    return {"status": "ok", "environment": "sql-repair-env", "version": "0.2.0"}


# ─── /faults ──────────────────────────────────────────────────────────────────
@app.get("/faults")
async def reveal_faults():
    """Reveal injected fault types — only meaningful after episode ends."""
    return {
        "injected_faults": env._injected_faults,
        "fault_count": len(env._injected_faults),
        "episode_done": env.state.step_count >= 20,
        "note": "Fault names revealed for post-episode analysis and curriculum tracking.",
    }


# ─── /curriculum ──────────────────────────────────────────────────────────────
@app.get("/curriculum")
async def get_curriculum():
    """Return current curriculum difficulty level and progression stats."""
    return curriculum.get_stats()


@app.post("/curriculum/reset")
async def reset_curriculum():
    """Reset curriculum to level 0. For testing only."""
    curriculum.reset()
    return {"message": "Curriculum reset to Novice level"}


# ─── /tasks ───────────────────────────────────────────────────────────────────
@app.get("/tasks")
async def list_tasks():
    """Return all tasks with descriptions and action schemas."""
    tasks_out = []
    for tid, task in TASKS.items():
        tasks_out.append({
            "id":           tid,
            "description":  task["description"],
            "difficulty":   task["difficulty"],
            "max_steps":    20,
            "action_schema": task["action_schema"],
            "subgoals": [
                {"name": name, "weight": weight}
                for name, weight in SUBGOALS[tid]
            ],
            "action_types": [
                "submit_query", "query_schema",
                "inspect_data", "list_tables", "run_test",
            ],
            "action_fields": {
                "action_type":  "str — one of the action_types above",
                "sql_query":    "str | None — SQL to execute (submit_query only)",
                "target_table": "str | None — table name (query_schema / inspect_data only)",
            },
        })
    return {"tasks": tasks_out, "count": len(tasks_out)}


# ─── /grader ──────────────────────────────────────────────────────────────────
class GraderRequest(BaseModel):
    episode_id: Optional[str] = None
    task_id: str = "easy"


@app.post("/grader")
async def grade_episode(req: GraderRequest = GraderRequest()):
    """Return the grader score for the current episode."""
    task_id = req.task_id
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}"
        )
    score = env.get_current_score()
    flags = env.get_achieved_flags()
    breakdown = {
        name: {"achieved": flags.get(name, False), "weight": weight}
        for name, weight in SUBGOALS[task_id]
    }
    return {
        "episode_id":       req.episode_id or env.state.episode_id,
        "task_id":          task_id,
        "score":            score,
        "max_score":        1.0,
        "subgoal_breakdown": breakdown,
    }


# ─── /baseline ────────────────────────────────────────────────────────────────
_baseline_lock = asyncio.Lock()
_baseline_cache: Optional[Dict[str, Any]] = None


@app.post("/baseline")
async def run_baseline(force: bool = False):
    """
    Trigger inference.py and return scores for all three tasks.
    Cached after first run — pass ?force=true to re-run.
    """
    global _baseline_cache

    if _baseline_cache is not None and not force:
        return _baseline_cache

    async with _baseline_lock:
        if _baseline_cache is not None and not force:
            return _baseline_cache

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "inference.py",
        )
        if not os.path.exists(script_path):
            raise HTTPException(status_code=500, detail="inference.py not found at repo root.")

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="inference.py timed out (300s).")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to run inference.py: {e}")

        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"inference.py exited {proc.returncode}. stderr: {stderr.decode()[:500]}",
            )

        try:
            result = json.loads(stdout.decode())
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"inference.py stdout is not valid JSON: {stdout.decode()[:300]}",
            )

        _baseline_cache = result
        return result


# ─── main() — required by openenv validate ────────────────────────────────────
def main():
    """
    CLI entry point for openenv_serve / uv run / python -m deployment modes.
    """
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()