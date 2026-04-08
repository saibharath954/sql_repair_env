#!/usr/bin/env python3
"""
inference.py — Baseline inference script for the SQL Repair Environment.

HACKATHON COMPLIANCE:
  - File is named 'inference.py' in the root directory  ✓
  - Uses the OpenAI Python client for all LLM calls      ✓
  - Reads credentials from environment variables:
      API_BASE_URL  — LLM endpoint (OpenAI-compatible)
      MODEL_NAME    — model identifier
      HF_TOKEN      — Hugging Face / API key

DEFAULT (free tier, no credit card):
    API_BASE_URL = https://generativelanguage.googleapis.com/v1beta/openai/
    MODEL_NAME   = gemini-2.5-flash
    HF_TOKEN     = your Google AI Studio API key  (free at aistudio.google.com)

ALTERNATIVELY — any OpenAI-compatible endpoint works:
    OpenAI:         API_BASE_URL=https://api.openai.com/v1  MODEL_NAME=gpt-4o-mini
    HF Inference:   API_BASE_URL=https://api-inference.huggingface.co/v1  MODEL_NAME=...
    Groq (free):    API_BASE_URL=https://api.groq.com/openai/v1  MODEL_NAME=llama-3.3-70b-versatile

Usage:
    # Gemini (free)
    export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
    export MODEL_NAME="gemini-2.5-flash"
    export HF_TOKEN="AIza..."          # your Google AI Studio key
    python inference.py

    # OpenAI
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="sk-..."
    python inference.py

Output (single JSON line to stdout):
    {"easy": 0.85, "medium": 0.52, "hard": 0.24, "metadata": {...}}
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─── configuration from environment variables (hackathon spec) ────────────────
API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/"   # Gemini default
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")   # API key — works for Gemini, OpenAI, HF, Groq

BASE_URL    = os.environ.get("BASE_URL", "http://localhost:7860").rstrip("/")
MAX_STEPS   = 18          # stay under environment's 20-step cap
MAX_RETRIES = 2
TASKS       = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert SQL engineer. You are interacting with a SQL Repair environment.
Your goal is to diagnose and fix a broken SQL query using the available tools.

Available action types (choose exactly one per step):
- list_tables      : See all tables in the database.
- query_schema     : Inspect the DDL/CREATE TABLE of a specific table. Provide target_table.
- inspect_data     : Preview the first 8 rows of a table. Provide target_table.
- submit_query     : Execute a SQL query. Provide sql_query. This is how you submit your fix.
- run_test         : Check your current score and sub-goal progress.

Strategy:
1. Start with list_tables to understand what tables exist.
2. Call query_schema on relevant tables to understand column names and types.
3. Inspect the broken_query shown in the initial hint — find the specific bug.
4. Submit a corrected query with submit_query.
5. If partial_score < 1.0, analyse the error_message and iterate.

The partial_score in each observation shows your progress (0.0 = nothing correct, 1.0 = perfect).
When done=true the episode ends. Maximise your score before that.

Always respond with a single valid JSON object. No markdown fences, no explanation text outside JSON."""


def _post(endpoint: str, payload: dict) -> dict:
    resp = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=30)
    resp.raise_for_status()
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


def run_task(client: OpenAI, task_id: str) -> float:
    """Run one full episode and return the final grader score."""
    obs = _post("/reset", {"task_id": task_id})
    history: List[str] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_message(obs, history)},
    ]
    final_score = 0.0
    last_raw = ""

    for step in range(MAX_STEPS):
        if obs.get("done"):
            break

        # ── LLM call via OpenAI client (endpoint configurable) ─────────────
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
        except Exception:
            action_dict = {"action_type": "list_tables"}

        action_type  = action_dict.get("action_type", "list_tables")
        sql_query    = action_dict.get("sql_query") or None
        target_table = action_dict.get("target_table") or None

        history.append(
            f"Step {step+1}: {action_type}"
            + (f" | {str(sql_query)[:80]}" if sql_query else "")
        )

        # ── Execute in environment ─────────────────────────────────────────
        try:
            obs = _post("/step", {
                "action_type":  action_type,
                "sql_query":    sql_query,
                "target_table": target_table,
            })
        except Exception as e:
            obs = {
                "error_message": str(e), "done": False,
                "partial_score": final_score, "step_count": step + 1,
            }

        final_score = obs.get("partial_score", final_score)
        messages.append({"role": "assistant", "content": last_raw})
        messages.append({"role": "user",      "content": _build_user_message(obs, history)})

        if obs.get("done"):
            break

    # ── Fetch authoritative grader score ──────────────────────────────────
    try:
        r = requests.post(
            f"{BASE_URL}/grader",
            params={"task_id": task_id},
            json={"task_id": task_id},
            timeout=10,
        )
        if r.ok:
            final_score = r.json().get("score", final_score)
    except Exception:
        pass

    return round(final_score, 4)


def main():
    if not HF_TOKEN:
        out = {
            "error": "HF_TOKEN environment variable not set. Set it to your API key.",
            "easy": 0.0, "medium": 0.0, "hard": 0.0,
        }
        print(json.dumps(out))
        sys.exit(1)

    # OpenAI client — base_url makes it work with Gemini, Groq, HF, or real OpenAI
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    scores: Dict[str, Any] = {}
    start = time.time()

    for task_id in TASKS:
        for attempt in range(MAX_RETRIES + 1):
            try:
                scores[task_id] = run_task(client, task_id)
                break
            except Exception:
                if attempt == MAX_RETRIES:
                    scores[task_id] = 0.0
                time.sleep(2)

    scores["metadata"] = {
        "model":              MODEL_NAME,
        "api_base_url":       API_BASE_URL,
        "base_url":           BASE_URL,
        "elapsed_seconds":    round(time.time() - start, 1),
        "max_steps_per_task": MAX_STEPS,
    }

    # Single JSON line to stdout — app.py /baseline endpoint parses this
    print(json.dumps(scores))


if __name__ == "__main__":
    main()