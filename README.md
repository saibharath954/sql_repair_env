# SQL Repair Environment

An OpenEnv reinforcement learning environment for training AI agents to repair broken SQL queries and clean dirty data — a real-world task performed by millions of data engineers and analysts every day.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Motivation

Every production data system has broken SQL: a missing comma in a SELECT, a JOIN on the wrong key, a GROUP BY that aggregates incorrectly, or an ETL pipeline that loaded TEXT instead of REAL. Fixing these errors requires multi-step reasoning — inspecting schemas, previewing data, forming hypotheses, and iterating on queries.

This environment provides a sandboxed, episodic interface where an LLM agent can practise exactly this skill with dense reward signals and deterministic programmatic graders.

---

## Environment Description

The agent interacts with an **in-memory SQLite database** that is freshly initialised on every `reset()` call. The database contains injected faults — syntax errors, wrong JOIN keys, dirty data — that the agent must diagnose and fix.

Each step the agent chooses one of five actions:

| Action type | Required fields | What it does |
|---|---|---|
| `list_tables` | — | Returns all table names in the DB |
| `query_schema` | `target_table` | Returns the DDL (`CREATE TABLE …`) for a table |
| `inspect_data` | `target_table` | Returns first 8 rows of a table |
| `submit_query` | `sql_query` | Executes a SQL string; returns rows or error |
| `run_test` | — | Runs the hidden correctness test; shows sub-goal breakdown |

---

## Action Space

```python
@dataclass
class SQLRepairAction(Action):
    action_type:  str            # Required. One of the five types above.
    sql_query:    Optional[str]  # Required for submit_query.
    target_table: Optional[str]  # Required for query_schema / inspect_data.
```

Example action (JSON):
```json
{"action_type": "submit_query", "sql_query": "SELECT name, salary FROM employees WHERE dept='Engineering' ORDER BY salary DESC"}
```

---

## Observation Space

```python
@dataclass
class SQLRepairObservation(Observation):
    query_result:     List[Dict]  # Rows returned by last SELECT
    error_message:    str         # Non-empty if last action raised an error
    schema_info:      str         # DDL string (after query_schema)
    rows_affected:    int         # Rows changed by INSERT/UPDATE/DELETE
    partial_score:    float       # Current grader score 0.0–1.0
    hint:             str         # Progressive hint (appears after 5/10/15 stuck steps)
    step_count:       int         # Steps taken in this episode
    tables:           List[str]   # Table names (after list_tables)
    task_id:          str         # Active task: easy | medium | hard
    task_description: str         # Human-readable goal
    broken_query:     str         # The defective query shown at episode start
    done:             bool        # True when episode ends
    reward:           float       # Dense step reward
```

---

## Tasks

### Easy — Syntax Error Fix
**Difficulty:** Easy | **Max steps:** 20

A salary report query fails because of a missing comma between column names in the SELECT clause.

```sql
-- Broken:
SELECT name salary FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC

-- Fixed:
SELECT name, salary FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC
```

**Grader sub-goals:**

| Sub-goal | Weight |
|---|---|
| Query executes without error | 0.25 |
| Correct columns returned | 0.25 |
| Correct row count | 0.25 |
| Correct values | 0.25 |

---

### Medium — Broken JOIN + Wrong GROUP BY
**Difficulty:** Medium | **Max steps:** 20

A sales dashboard query joins on a non-existent column and groups by the wrong field. The agent must inspect the schema of three tables to identify and fix both bugs.

```sql
-- Broken:
SELECT p.category, SUM(p.unit_price * o.quantity) AS revenue
FROM orders o
JOIN products p ON o.product_name = p.product_name   -- wrong: o has no product_name
GROUP BY o.product_id                                   -- wrong: should be p.category
ORDER BY revenue DESC

-- Fixed:
SELECT p.category, SUM(p.unit_price * o.quantity) AS revenue
FROM orders o JOIN products p ON o.product_id = p.product_id
GROUP BY p.category ORDER BY revenue DESC
```

**Grader sub-goals:**

| Sub-goal | Weight |
|---|---|
| Schema inspected (query_schema called) | 0.15 |
| Query executes without error | 0.20 |
| Correct columns returned | 0.15 |
| Correct row count | 0.20 |
| Correct values (exact revenue figures) | 0.30 |

---

### Hard — Dirty ETL Data Cleaning + Analytical Query
**Difficulty:** Hard | **Max steps:** 20

A nightly ETL job loaded transaction data with three problems: duplicate rows, orphan foreign keys (customer_id=99 doesn't exist), and the `amount` column stored as TEXT instead of REAL. The agent must detect all three issues and write a query that produces the correct per-customer spend totals.

**Grader sub-goals:**

| Sub-goal | Weight |
|---|---|
| Duplicates detected (inspect_data reveals them) | 0.15 |
| Type cast present (`CAST(amount AS REAL)` in SQL) | 0.15 |
| Invalid FK excluded (no customer_id=99 in result) | 0.10 |
| Correct row count | 0.20 |
| Correct values (exact spend totals) | 0.40 |

---

## Reward Function

Dense, potential-based reward shaping:

```
R(t) = Φ(s_t) − Φ(s_{t-1}) − 0.02
```

Where `Φ(s)` is the weighted sum of achieved sub-goals (the grader score). Addional adjustments:

- **+1.0** terminal bonus when `partial_score == 1.0`
- **−0.50** for destructive SQL (DROP TABLE, TRUNCATE)
- **−0.15** for repeating SQL errors 3+ times in a row

---

## Baseline Scores

Measured using `gpt-4o-mini` at temperature 0, max 18 steps per task:

| Task | Score | Notes |
|---|---|---|
| easy | 0.85 | Solves reliably; occasional off-by-one on ORDER BY |
| medium | 0.52 | Usually fixes JOIN key; sometimes misses GROUP BY |
| hard | 0.24 | Rarely finds all three data issues in one episode |

Run the baseline yourself:
```bash
OPENAI_API_KEY=sk-... BASE_URL=http://localhost:7860 python baseline_inference.py
```

---

## Setup & Usage

### Local development (no Docker)

```bash
git clone https://github.com/YOUR_USERNAME/sql-repair-env
cd sql-repair-env
pip install -e ".[dev]"

# Run the server
uvicorn server.app:app --reload --port 7860

# In another terminal, call reset
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'

# List tasks
curl http://localhost:7860/tasks

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "list_tables"}'
```

### Docker

```bash
docker build -t sql-repair-env -f server/Dockerfile .
docker run -p 7860:7860 sql-repair-env

# Health check
curl http://localhost:7860/health
```

### Python client

```python
from sql_repair_env import SQLRepairEnv, SQLRepairAction

with SQLRepairEnv(base_url="https://YOUR_SPACE.hf.space").sync() as env:
    obs = env.reset(task_id="medium")
    print(obs.observation.hint)

    result = env.step(SQLRepairAction(action_type="list_tables"))
    print(result.observation.tables)
```

### Run tests

```bash
pytest test_environment.py -v
```

---

## Custom Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/tasks` | GET | List all tasks with action schemas |
| `/grader` | POST | Score the current episode |
| `/baseline` | POST | Run baseline inference; returns scores JSON |

---

## Project Structure

```
sql_repair_env/
├── __init__.py              # Package exports
├── models.py                # SQLRepairAction, SQLRepairObservation (Pydantic)
├── client.py                # SQLRepairEnv (EnvClient subclass)
├── baseline_inference.py    # OpenAI agent loop → JSON scores
├── test_environment.py      # 20+ unit tests
├── openenv.yaml             # Environment manifest
├── pyproject.toml           # Package metadata
└── server/
    ├── __init__.py
    ├── sql_repair_environment.py  # Core Environment logic
    ├── app.py                     # FastAPI + custom endpoints
    ├── tasks.py                   # Task definitions
    ├── grader.py                  # Deterministic scorer
    ├── requirements.txt
    └── Dockerfile
```

---

## Deploy to Hugging Face Spaces

```bash
pip install openenv-core
openenv push --repo-id YOUR_USERNAME/sql-repair-env
```

Then submit your Space URL in the hackathon dashboard.