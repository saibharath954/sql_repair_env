#!/usr/bin/env python3
"""
inference.py — Baseline inference script for the SQL Repair Environment.

HACKATHON COMPLIANCE:
  - File named 'inference.py' in root directory                      ✓
  - Uses OpenAI Python client for all LLM calls                      ✓
  - Emits [START], [STEP], [END] structured stdout logs              ✓
  - Reads credentials from:
      API_BASE_URL  — LLM endpoint (OpenAI-compatible)
      MODEL_NAME    — model identifier
      HF_TOKEN      — API key

DEFAULT (free):
    API_BASE_URL = https://generativelanguage.googleapis.com/v1beta/openai/
    MODEL_NAME   = gemini-2.5-flash
    HF_TOKEN     = your Google AI Studio key (aistudio.google.com)

Usage:
    export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
    export MODEL_NAME="gemini-2.5-flash"
    export HF_TOKEN="AIza..."
    python inference.py
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─── environment variables (hackathon spec) ───────────────────────────────────
API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
BASE_URL   = os.environ.get("BASE_URL", "http://localhost:7860").rstrip("/")
MAX_STEPS  = 18
MAX_RETRIES = 2
TASKS      = ["easy", "medium", "hard"]
BENCHMARK  = "sql-repair-env"


# ─── mandatory structured log helpers ────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    parts = [
        "[STEP]",
        f"step={step}",
        f"action={json.dumps(action)}",
        f"reward={reward:.4f}",
        f"done={done}",
    ]
    if error:
        parts.append(f"error={json.dumps(error)}")
    print(" ".join(parts), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={success} steps={steps} score={score:.4f} "
        f"rewards={json.dumps([round(r, 4) for r in rewards])}",
        flush=True,
    )


# ─── system prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert SQL engineer fixing broken SQL queries.

Available action types (choose exactly one per step):
- list_tables      : See all tables in the database.
- query_schema     : Inspect the DDL of a table. Provide target_table.
- inspect_data     : Preview first 8 rows of a table. Provide target_table.
- submit_query     : Execute a SQL query. Provide sql_query.
- run_test         : Check current score and sub-goal progress.

Strategy:
1. list_tables → understand available tables.
2. query_schema → understand column names and types.
3. Identify the bug in broken_query (shown in hint).
4. submit_query with the corrected SQL.
5. If partial_score < 1.0, analyse error_message and iterate.

partial_score shows progress (0.0=nothing, 1.0=perfect). Maximise before done=true.

Respond ONLY with a single valid JSON object. No markdown, no extra text."""


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> dict:
    resp = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=30)
    if not resp.ok:
        # This will expose exactly which field Pydantic is rejecting!
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def _build_user_message(obs: dict, history: List[str]) -> str:
    parts = []
    if obs.get("hint"):
        parts.append(f"OBSERVATION:\n{obs['hint']}")
    if obs.get("error_message"):
        parts.append(f"ERROR: {obs['error_message']}")
    if obs.get("schema_info"):
        parts.append(f"SCHEMA:\n{obs['schema_info']}")
    if obs.get("tables"):
        parts.append(f"TABLES: {obs['tables']}")
    if obs.get("query_result"):
        parts.append(
            f"QUERY RESULT ({len(obs['query_result'])} rows):\n"
            + json.dumps(obs["query_result"][:10], indent=2)
        )
    parts.append(
        f"partial_score={obs.get('partial_score', 0):.2f}  "
        f"step={obs.get('step_count', 0)}/20  done={obs.get('done', False)}"
    )
    if history:
        parts.append("PREVIOUS ACTIONS (last 4):\n" + "\n".join(history[-4:]))
    return "\n\n".join(parts)


# ─── single task episode ──────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> float:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    reset_resp = _post("/reset", {"task_id": task_id})
    obs = reset_resp.get("observation", reset_resp)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_message(obs, history)},
    ]
    last_raw = ""

    for step in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            break

        # LLM call (OpenAI client — endpoint set via API_BASE_URL)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )
            last_raw = response.choices[0].message.content or "{}"
            action_dict = json.loads(last_raw)
        except json.JSONDecodeError:
            action_dict = {"action_type": "list_tables"}
            last_raw = json.dumps(action_dict)
        except Exception as exc:
            print(f"[DEBUG] LLM call failed: {exc}", flush=True)
            action_dict = {"action_type": "list_tables"}
            last_raw = json.dumps(action_dict)

        action_type  = action_dict.get("action_type", "list_tables")
        sql_query    = action_dict.get("sql_query") or None
        target_table = action_dict.get("target_table") or None
        action_label = action_type + (f":{str(sql_query)[:60]}" if sql_query else "")
        history.append(f"Step {step}: {action_label}")

        step_error: Optional[str] = None
        reward = 0.0
        done = False
        try:
            # 3. CONSTRUCT THE ACTION PAYLOAD
            action_payload = {"action_type": action_type}
            if sql_query is not None:
                action_payload["sql_query"] = sql_query
            if target_table is not None:
                action_payload["target_table"] = target_table

            # 4. WRAP IN THE "action" KEY REQUIRED BY FASTAPI/OPENENV
            step_resp = _post("/step", {"action": action_payload})
            obs    = step_resp.get("observation", step_resp)
            reward = step_resp.get("reward", 0.0)
            done   = step_resp.get("done", obs.get("done", False))
            if obs.get("error_message"):
                step_error = obs["error_message"]
        except Exception as exc:
            step_error = str(exc)
            obs = {"error_message": step_error, "done": False,
                   "partial_score": score, "step_count": step}

        score = obs.get("partial_score", score)
        rewards.append(reward)
        steps_taken = step

        # Mandatory [STEP] log
        log_step(step=step, action=action_label, reward=reward, done=done, error=step_error)

        messages.append({"role": "assistant", "content": last_raw})
        messages.append({"role": "user",      "content": _build_user_message(obs, history)})

        if done:
            break

    # Authoritative grader score
    try:
        r = requests.post(
            f"{BASE_URL}/grader",
            params={"task_id": task_id},
            json={"task_id": task_id},
            timeout=10,
        )
        if r.ok:
            score = r.json().get("score", score)
    except Exception:
        pass

    final_score = min(max(score, 0.0), 1.0)
    success = final_score >= 0.5

    # Mandatory [END] log
    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
    return round(final_score, 4)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if not HF_TOKEN:
        out = {"error": "HF_TOKEN not set", "easy": 0.0, "medium": 0.0, "hard": 0.0}
        print(json.dumps(out))
        sys.exit(1)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    scores: Dict[str, Any] = {}
    start = time.time()

    for task_id in TASKS:
        for attempt in range(MAX_RETRIES + 1):
            try:
                scores[task_id] = run_task(client, task_id)
                break
            except Exception as exc:
                print(f"[DEBUG] task={task_id} attempt={attempt} error={exc}", flush=True)
                if attempt == MAX_RETRIES:
                    scores[task_id] = 0.0
                time.sleep(2)

    scores["metadata"] = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "base_url": BASE_URL,
        "elapsed_seconds": round(time.time() - start, 1),
        "max_steps_per_task": MAX_STEPS,
    }

    # Single JSON line to stdout
    print(json.dumps(scores))


if __name__ == "__main__":
    main()