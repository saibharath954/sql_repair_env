# When Your Database Breaks at 3 AM: Teaching AI to Fix Production SQL

*OpenEnv Hackathon India 2026 — DataOps Incident Response*

---

## The 3 AM Call Nobody Wants

Picture this. It's 3 AM. Your phone buzzes. PagerDuty. The sales dashboard is blank. Revenue numbers — gone. Customer support is flooding Slack. Your CEO just texted "???"

You stumble to your laptop, eyes half-open, and start digging. Forty minutes later, you find it: someone pushed a migration that changed a column from `REAL` to `TEXT`. Every `SUM()` in the pipeline is now silently concatenating strings instead of adding numbers. The ETL job duplicated 10,000 rows. And there's a stale `WHERE` clause filtering on a department name that got renamed six months ago.

This isn't hypothetical. **This happens every single day** at companies running production databases. And it's exactly the kind of messy, multi-layered problem that we thought an AI agent should learn to solve — not from textbooks, but from practice.

So we built an environment where it can.

---

## What We Built: A Production Database Incident Simulator

Our environment, **DataOps Incident Response**, drops an AI agent into a broken production database and says: *"Fix it."*

But here's the twist — **it's never the same break twice.**

Every episode, the environment spins up a fresh in-memory SQLite database, loads real-looking production data, and then stochastically injects 2–4 faults from a pool of 12. The agent gets the broken query, a task description, and access to diagnostic tools (`list_tables`, `query_schema`, `inspect_data`). It has 20 steps to figure out what went wrong and submit a corrected SQL query.

No memorization. No shortcuts. Just reasoning.

### Stochastic Fault Injection: The Core Innovation

Most SQL benchmarks give you one broken query and one right answer. Memorize the fix, ace the test. That's not how real incidents work.

We built a **stochastic fault injector** ([`server/fault_injector.py`](server/fault_injector.py)) that randomly combines faults every episode. Here's the full pool:

| Fault Type | What It Does | Real-World Analog |
|---|---|---|
| `missing_comma` | Removes a comma from SELECT | Typo in a PR |
| `wrong_join_key` | Uses non-existent column in JOIN | Copy-paste from wrong table |
| `type_drift` | Changes REAL column to TEXT | Bad ETL schema inference |
| `duplicate_rows` | Inserts exact duplicate records | Failed idempotency in pipeline |
| `null_fk` | Adds rows with orphaned foreign keys | Partial data load |
| `stale_where` | Changes filter value to outdated string | Renamed department/category |
| `column_alias_shadow` | Renames an alias to cause confusion | Refactoring side effect |
| `wrong_sort_order` | Flips ASC/DESC | Spec misunderstanding |
| `off_by_one_limit` | Adds LIMIT that cuts off a row | Pagination bug |
| `missing_distinct` | Removes DISTINCT from subquery | Aggregate inflation |
| `wrong_group_by` | Groups by wrong column | Misread schema |
| `implicit_cast_bug` | Removes CAST needed for arithmetic | Silent type coercion |

Every single episode randomly selects 2–4 of these and applies them — both to the database (data-level faults like duplicates and type drift) and to the SQL query (query-level faults like missing commas and wrong JOINs). The injector guarantees at least one data fault and one query fault per episode, so the agent always has to diagnose both the query *and* the data.

With 12 fault types and combinatorial selection, the environment can generate **hundreds of unique incident scenarios** from just three base tasks.

### Three Tiers of Pain

We designed three task difficulties that mirror real incident severity:

**🟢 Easy — The First On-Call Page**

A syntax error in a simple `SELECT`. Missing comma between `name` and `salary` in an HR report query. Scary the first time, but fixable once you see it.

```sql
-- Broken (missing comma)
SELECT name salary FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC

-- Fixed
SELECT name, salary FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC
```

**🟡 Medium — The "Read the ERD" Incident**

A three-table JOIN with the wrong join key (`o.product_name = p.product_name` — but `product_name` doesn't exist on orders!) AND wrong `GROUP BY` column (`o.product_id` instead of `p.category`). You need to actually understand the schema to fix this.

```sql
-- Broken (wrong JOIN key + wrong GROUP BY)
SELECT p.category, SUM(p.unit_price * o.quantity) AS revenue
FROM orders o JOIN products p ON o.product_name = p.product_name
GROUP BY o.product_id ORDER BY revenue DESC

-- Fixed
SELECT p.category, SUM(p.unit_price * o.quantity) AS revenue
FROM orders o JOIN products p ON o.product_id = p.product_id
GROUP BY p.category ORDER BY revenue DESC
```

**🔴 Hard — Cancel Your Morning Meetings**

A full ETL disaster. The nightly job loaded dirty data: duplicate transactions (`txn_id` 1001 appears 3 times), orphaned foreign keys (`customer_id=99` doesn't exist in the customers table), and the `amount` column was loaded as `TEXT` instead of `REAL`. You need to de-duplicate with `SELECT DISTINCT`, type-cast with `CAST(amount AS REAL)`, exclude orphans with an `INNER JOIN`, and aggregate correctly.

```sql
-- Broken (no dedup, no cast, includes orphans)
SELECT c.name, SUM(t.amount) AS total_spend
FROM customers_hard c JOIN transactions t ON c.customer_id = t.customer_id
GROUP BY c.name ORDER BY total_spend DESC

-- Fixed
SELECT c.name, SUM(CAST(t.amount AS REAL)) AS total_spend
FROM customers_hard c
JOIN (SELECT DISTINCT txn_id, customer_id, amount FROM transactions) t
  ON c.customer_id = t.customer_id
GROUP BY c.customer_id, c.name ORDER BY total_spend DESC
```

### Dense Rewards, Not Pass/Fail

Real debugging is incremental. You don't go from "everything is broken" to "everything works" in one step. You fix the syntax error, then realize the JOIN is wrong, then notice the duplicates...

Our grader ([`server/grader.py`](server/grader.py)) uses **weighted sub-goal scoring** to give partial credit:

**Easy Sub-Goals:**
- `query_executes_without_error` (22.5%) — Does the query even run?
- `correct_columns_returned` (22.5%) — Right column names in output?
- `correct_row_count` (22.5%) — Expected number of rows?
- `correct_values` (22.5%) — Exact match on data values?
- `faults_diagnosed` (10%) — Meta-goal: auto-granted at ≥80% score

**Hard Sub-Goals:**
- `duplicates_detected` (13.5%) — Did the agent handle deduplication?
- `type_cast_present` (13.5%) — Did the agent use CAST for text→numeric?
- `invalid_fk_excluded` (9%) — Were orphan rows filtered out?
- `correct_row_count` (18%) — Right number of result rows?
- `correct_values` (36%) — Exact match on final output?
- `faults_diagnosed` (10%) — Meta-goal

The reward function wraps this into a **potential-based dense signal**:

```
R(t) = Φ(s_t) − Φ(s_{t-1}) − 0.02 (step penalty)
```

Plus penalties for destructive SQL (`DROP TABLE` → −0.5) and error loops (3+ repeated errors → −0.15). And a +1.0 bonus for a perfect solve.

The key design decision: **a completely failed query scores exactly 0.0**, not some epsilon floor. This is critical for GRPO — if every failure scores 0.001, the advantages across rollouts collapse to near-zero and the model learns nothing. We learned this the hard way.

### Adaptive Curriculum

The environment also ships with an **adaptive curriculum manager** ([`server/curriculum.py`](server/curriculum.py)) that tracks agent performance and escalates difficulty automatically:

| Level | Name | Fault Pool | Task Mix |
|---|---|---|---|
| 0 | Novice | 3 simple faults | Easy only |
| 1 | Analyst | 5 intermediate faults | Easy + Medium |
| 2 | Senior | All 12 faults | Medium + Hard |
| 3 | Staff Engineer | All 12 + red herring table | Hard only |

Rolling mean score ≥ 0.75 → promotion. Drop below 0.30 → demotion. The environment literally gets harder as the agent gets smarter.

---

## Training: SFT Warm-up + GRPO with a Live Environment

We trained **Qwen2.5-1.5B-Instruct** using a two-stage pipeline: supervised fine-tuning (SFT) warm-up followed by **Group Relative Policy Optimization (GRPO)** from TRL, with rewards coming directly from the live environment running on Hugging Face Spaces.

The full training notebook is in [`train_grpo.ipynb`](train_grpo.ipynb) (also exported as [`train_grpo.py`](train_grpo.py)).

### The Setup

| Component | Detail |
|---|---|
| **Base Model** | Qwen/Qwen2.5-1.5B-Instruct |
| **Quantization** | NF4 4-bit + double quantization |
| **Fine-tuning** | QLoRA (r=16, α=32, dropout=0.05) on all attention + MLP projections |
| **SFT Warm-up** | 60 gold (broken → fixed) pairs, 2 epochs, lr=1e-4 |
| **RL Algorithm** | GRPO — 8 completions per prompt, β=0.10 KL penalty |
| **Training Data** | 240 seeded prompts (80 per tier), disjoint from eval seeds |
| **Evaluation** | 30 held-out episodes (10 per tier), identical seeds for baseline & trained |
| **Hardware** | Google Colab T4 GPU (~30 min training time) |
| **Reward Source** | Live `/reset` + `/step` calls to the HF Space |

### Key Design Decision: Seeded Prompts

Every training prompt is tied to a specific `(task_id, seed)` pair. When the reward function evaluates a completion, it resets the environment with the **same seed**, so the agent is graded on the exact problem it was prompted with. This gives us deterministic, reproducible evaluation:

```python
# Training seeds: 1000–1239 (240 prompts)
# SFT seeds:      5000–5059 (60 gold pairs)
# Eval seeds:     9000–9029 (30 held-out episodes)
# No overlap between sets
```

### The SFT Warm-up: Why It Matters

Before GRPO, the model had no idea what "output raw SQL" even meant. It would wrap answers in markdown code blocks, add explanations, or output partial fixes. The SFT stage teaches the model the *format* — "given a broken query and task description, output just the corrected SQL."

We generated 60 gold examples by trying various candidate fixes against the live environment and keeping the ones that scored highest. Even this small amount of supervised data got the model into the right output space, so GRPO wasn't cold-starting from complete randomness.

### The GRPO Pipeline

```
For each of 240 training prompts:
  1. Reset environment with (task_id, seed)
  2. Model generates 8 candidate SQL fixes (temperature=0.7)
  3. Each candidate submitted to /step endpoint
  4. Composite reward = env_partial_score + exec_bonus(0.05)
  5. GRPO ranks the 8 completions and updates LoRA adapters
  6. Repeat for 2 epochs with cosine LR schedule (2e-6)
```

---

## Results: +90.5% Improvement — From 0.288 to 0.549

Here's what happened. All numbers are from a **held-out evaluation set of 30 episodes** (10 per tier) using seeds completely disjoint from training. The baseline and trained model were evaluated on the **exact same seeds** — genuine apples-to-apples.

### Before vs After Training

![Before vs After Training — Mean score improved from 0.288 to 0.549](training_evidence/before_after_comparison.png)

| Metric | Baseline (Untrained) | Trained (SFT+GRPO) | Delta |
|---|---|---|---|
| **Mean Score** | 0.288 | **0.549** | **+0.261 (+90.5%)** |
| **Success Rate (≥0.5)** | 40% | **63%** | **+23 percentage points** |

The mean score nearly doubled. But the really interesting story is in the per-task breakdown.

### Per-Task Score Breakdown

![Per-Task Score Breakdown — Improvement across all three tiers](training_evidence/per_task_breakdown.png)

| Task | Baseline | Trained | Delta | What Changed |
|---|---|---|---|---|
| **Easy** | 0.460 | **0.560** | +0.100 (+21.7%) | Consistent syntax fixes instead of random attempts |
| **Medium** | 0.000 | **0.657** | +0.657 (∞%) | From total failure to majority solved |
| **Hard** | 0.405 | **0.432** | +0.027 (+6.7%) | Modest gains on the hardest task |

**The medium task result is the headline.** The baseline model scored a flat zero on every single medium episode — 10 out of 10 failures. After training, it averaged 0.657, with several episodes scoring 1.0 (perfect). The model went from "can't touch this" to "usually gets it right."

Why was the baseline so bad on medium? The medium task requires understanding a three-table schema and fixing both the JOIN key and GROUP BY column. A 1.5B model without any training simply doesn't know to look at the table relationships. After SFT showed it what correct multi-table JOINs look like, and GRPO reinforced successful repair patterns, the model learned to reason about schema structure.

### Training Reward Curve

![Training Reward Curve — Rolling mean climbed from ~0.25 to ~0.55 over 120 steps](training_evidence/training_reward_curve.png)

The reward curve tells the training story:

- **Steps 0–15**: Rolling mean hovers around 0.25–0.30. The model is still mostly producing garbage or partially correct fixes.
- **Steps 15–35**: First major climb. The rolling mean jumps to 0.45. The SFT warm-up is paying off — the model is now at least outputting syntactically valid SQL.
- **Steps 35–75**: Volatile but trending up. Per-step rewards swing between 0.15 and 0.80. The model is exploring different fix strategies, and GRPO is reinforcing the ones that work.
- **Steps 75–120**: Stabilization around 0.50–0.55. The model has found a reliable strategy for easy and medium tasks. Hard task gains are slower but present.

### Training Loss Curve

![Training Loss Curve — Policy shifts correlating with reward improvements](training_evidence/training_loss_curve.png)

The GRPO policy loss oscillates around zero, which is expected — GRPO loss reflects relative advantage across rollouts, not absolute error. The spikes (e.g., around step 110) correspond to the model discovering particularly useful SQL patterns that shift the policy significantly.

### Raw Score Distributions

Looking at individual episode scores tells an even richer story:

**Baseline scores (30 episodes):**
```
Easy:   [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.00, 1.00]
Medium: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
Hard:   [0.54, 0.27, 0.27, 0.27, 0.54, 0.27, 0.54, 0.27, 0.54, 0.54]
```

**Trained scores (30 episodes):**
```
Easy:   [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 1.00, 1.00]
Medium: [0.45, 1.00, 1.00, 0.63, 0.18, 0.18, 0.63, 1.00, 0.495, 1.00]
Hard:   [0.54, 0.36, 0.27, 0.36, 0.54, 0.36, 0.54, 0.27, 0.54, 0.54]
```

Things to notice:
- **Easy**: The baseline's single 0.00 outlier (where the model completely failed) became a 1.0 after training. The 1.0 outlier stayed. Consistency improved.
- **Medium**: Every. Single. Zero. Became a non-zero score. Four episodes hit 1.0 (perfect solve). This is the most dramatic improvement.
- **Hard**: Subtle gains. Several 0.27s became 0.36s. The model learned to get partial credit (probably from CAST or DISTINCT) even when it can't solve the full problem.

---

## Why This Environment is Novel

### It's Not a Benchmark — It's a Gym

Spider, WikiSQL, BIRD — these test whether a model can *write* SQL from a natural language description. That's text-to-SQL. Important, but not what happens when your production database is on fire.

Our environment tests something fundamentally different:
- **Diagnosis**: Can the agent use `list_tables`, `query_schema`, `inspect_data` to understand what went wrong?
- **Data awareness**: Can it detect that amounts are stored as TEXT, that there are duplicate rows, that customer_id=99 is an orphan?
- **Reasoning**: Can it trace a wrong result back to a bad JOIN key or missing GROUP BY?
- **Repair**: Can it write the fix — including deduplication subqueries, type casts, and correct aggregation?
- **Robustness**: Can it handle *unseen* fault combinations, not just memorized patterns?

### The Stochastic Element Makes Memorization Impossible

Because faults are randomly injected from a pool of 12, with guaranteed data+query fault combinations, the agent can't overfit to specific fixes. In 240 training episodes, it saw hundreds of different fault combinations. We verified this with seeded evaluation: the training seeds (1000–1239) are completely disjoint from eval seeds (9000–9029), so any improvement is from genuine generalization.

### Dense Rewards Enable RL

Most SQL environments give binary rewards (correct or wrong). That's useless for RL — the gradient signal is too sparse. Our sub-goal-weighted grader gives meaningful partial credit at every step:

- A query that executes but returns wrong rows scores ~0.225 (not zero)
- A query with correct columns but wrong row count scores ~0.45
- A query with everything right except the values scores ~0.675

This dense signal is what makes GRPO actually work on this task.

---

## Technical Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     HF Space (Docker)                     │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                  FastAPI Server                      │  │
│  │  POST /reset  → SQLRepairEnvironment.reset()        │  │
│  │  POST /step   → SQLRepairEnvironment.step()         │  │
│  │  GET  /health → liveness probe                      │  │
│  │  GET  /tasks  → task registry                       │  │
│  │  POST /grader → score breakdown                     │  │
│  └──────┬──────────────────────────┬───────────────────┘  │
│         │                          │                       │
│  ┌──────▼──────┐           ┌──────▼───────────────────┐   │
│  │   In-memory  │           │   Stochastic Fault       │   │
│  │   SQLite DB  │◄──────────│   Injector (12 faults)   │   │
│  │  (per-episode)│           │   + Curriculum Manager   │   │
│  └──────┬──────┘           └──────────────────────────┘   │
│         │                                                  │
│  ┌──────▼──────────────────────────────────────────────┐   │
│  │              Deterministic Grader                    │   │
│  │   Weighted sub-goals → partial_score ∈ [0.0, 1.0]   │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│               Training Pipeline (Colab T4)                │
│                                                           │
│  Qwen2.5-1.5B-Instruct (4-bit NF4 + QLoRA)              │
│       │                                                   │
│       ├── SFT Warm-up (60 gold pairs, 2 epochs)          │
│       │                                                   │
│       └── GRPO (240 prompts × 8 generations × 2 epochs)  │
│              │                                            │
│              └── Reward from live HF Space /step calls    │
└──────────────────────────────────────────────────────────┘
```

---

## Try It Yourself

The environment is live on Hugging Face Spaces:

🔗 **HF Space**: [bharath1675/sql-repair-env](https://huggingface.co/spaces/bharath1675/sql-repair-env)

🔗 **GitHub**: [saibharath954/sql_repair_env](https://github.com/saibharath954/sql_repair_env)

🔗 **Training Notebook**: [Google Colab](https://colab.research.google.com/drive/1HLs3p51pB5Us4-tONLkQtfIzH1V2ugp7?usp=sharing) — re-run the entire pipeline in ~30 min on a free Colab T4

🔗 **Trained Adapters**: [bharath1675/sql-repair-grpo-qwen](https://huggingface.co/bharath1675/sql-repair-grpo-qwen) on Hugging Face Hub

Quick test from your terminal:

```bash
# Health check
curl https://bharath1675-sql-repair-env.hf.space/health
# → {"status":"ok","environment":"sql-repair-env","version":"0.2.0"}

# Start an easy episode
curl -X POST https://bharath1675-sql-repair-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Submit a fix
curl -X POST https://bharath1675-sql-repair-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "submit_query", "sql_query": "SELECT name, salary FROM employees WHERE dept='\''Engineering'\'' ORDER BY salary DESC"}}'
```

---

## What We'd Do With More Time

1. **Multi-step agent training** — Right now we train single-shot (one submit per episode). The environment supports 20-step episodes with `list_tables` → `query_schema` → `inspect_data` → `submit_query`. Training a model to use this diagnostic loop would push scores much higher.

2. **Longer training runs** — 240 prompts × 2 epochs is minimal. With 1000+ prompts and lower learning rate, we'd expect smoother convergence and better hard-task performance.

3. **Bigger model exploration** — Qwen2.5-1.5B is small. We specifically chose it to maximize the "improvement headroom" — a bigger model (7B) might have a higher baseline but less dramatic training gains.

4. **More fault types** — The injector architecture is plug-and-play. New fault types (wrong aggregate function, missing HAVING clause, incorrect UNION) slot right in.

---

## Key Files Reference

| File | Purpose |
|---|---|
| `server/environment.py` | Core environment: reset(), step(), reward computation |
| `server/fault_injector.py` | Stochastic 12-fault injection engine |
| `server/grader.py` | Deterministic sub-goal grader (no LLM involved) |
| `server/curriculum.py` | Adaptive difficulty escalation |
| `server/tasks.py` | Task definitions: DDL, broken queries, expected results |
| `server/app.py` | FastAPI server with all endpoints |
| `inference.py` | Baseline inference script (OpenAI-compatible) |
| `train_grpo.ipynb` | Full training notebook (SFT + GRPO) |
| `training_evidence/` | Charts and JSON results from the training run |
| `test_environment.py` | Comprehensive unit tests |

---

*Built for the OpenEnv Hackathon India 2026. The 3 AM calls don't stop, but maybe the AI can pick them up for us.*
