---
title: SQL Repair Env
emoji: 🗄️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# SQL Repair Environment

An [OpenEnv](https://huggingface.co/openenv) environment for training AI agents to repair broken SQL queries and clean dirty data.

## Overview

Agents interact with an isolated **SQLite** database via tool-style actions and receive **dense partial-credit rewards**. Three tasks span easy → medium → hard difficulty, covering real data-engineering failure modes: syntax errors, broken JOINs, dirty ETL data, type mismatches, and FK violations.

## Environment Description

### Action Space

`SQLRepairAction` with three fields:

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | One of: `submit_query`, `query_schema`, `inspect_data`, `list_tables`, `run_test` |
| `sql_query` | `str \| None` | SQL to execute. Required for `submit_query`. |
| `target_table` | `str \| None` | Table name. Required for `query_schema` / `inspect_data`. |

### Observation Space

`SQLRepairObservation` with fields:

| Field | Type | Description |
|---|---|---|
| `query_result` | `list[dict]` | Rows returned by last SELECT |
| `error_message` | `str` | Non-empty if last action raised an error |
| `schema_info` | `str` | DDL after `query_schema` |
| `rows_affected` | `int` | Count of rows changed by DML |
| `partial_score` | `float` | Current grader score 0.0–1.0 |
| `hint` | `str` | Progressive hint after stuck steps |
| `step_count` | `int` | Steps taken in this episode |
| `tables` | `list[str]` | Table names after `list_tables` |
| `task_id` | `str` | Active task |
| `task_description` | `str` | Human-readable goal |
| `broken_query` | `str` | Original broken SQL |

### Reward Function

Dense, potential-based:
```
R(t) = Φ(s_t) - Φ(s_{t-1}) - 0.02 (step penalty)
```
- **+1.0** terminal bonus on perfect score (1.0)
- **−0.5** penalty for destructive SQL (DROP/TRUNCATE/DELETE without WHERE)
- **−0.15** penalty for repeated syntax errors (≥3 in a row)

## Tasks

### Easy — Missing Comma
Fix a single syntax error (missing comma between column names) in an HR salary report query.
- **Score 1.0**: correct columns, correct row count, correct values

### Medium — Broken JOIN + Wrong GROUP BY
Fix a sales dashboard query with a wrong JOIN key (`o.product_name` → `o.product_id`) and incorrect `GROUP BY` column (`o.product_id` → `p.category`).
- **Score 1.0**: schema inspected, correct JOIN, correct aggregation

### Hard — Dirty ETL Data
The ETL loaded dirty data: duplicate transactions, invalid FK references (customer_id=99), and `amount` stored as TEXT. Write a query that de-duplicates, type-casts, excludes orphans, and returns correct per-customer totals.
- **Score 1.0**: deduplication, type cast, FK exclusion, correct values

## Setup

### Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/sql-repair-env
cd sql-repair-env
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest test_environment.py -v

# Start server
uvicorn server.app:app --port 7860
```

### Docker

```bash
docker build -t sql-repair-env .
docker run -p 7860:7860 \
  -e HF_TOKEN="your_api_key" \
  -e MODEL_NAME="gemini-2.5-flash" \
  sql-repair-env

# Verify
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

### Running the Baseline Inference Script

```bash
# Free tier — Google AI Studio (get key at aistudio.google.com)
export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL_NAME="gemini-2.5-flash"
export HF_TOKEN="AIza..."
export BASE_URL="http://localhost:7860"

python inference.py
```

Output:
```json
{"easy": 0.85, "medium": 0.52, "hard": 0.24, "metadata": {...}}
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM endpoint (OpenAI-compatible) | Gemini endpoint |
| `MODEL_NAME` | Model identifier | `gemini-2.5-flash` |
| `HF_TOKEN` | API key (Gemini/OpenAI/Groq/HF) | — |
| `BASE_URL` | Environment server URL | `http://localhost:7860` |
| `PORT` | Server port | `7860` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode. Body: `{"task_id": "easy"}` |
| `POST` | `/step` | Execute action |
| `GET` | `/state` | Episode metadata |
| `GET` | `/health` | Liveness probe |
| `GET` | `/tasks` | All tasks + action schemas |
| `POST` | `/grader` | Score for current episode |
| `POST` | `/baseline` | Run inference.py, return scores |

## License

MIT