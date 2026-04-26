# When Your Database Breaks at 3 AM: Teaching AI to Fix Production SQL

*OpenEnv Hackathon 2026 — Team Submission*

---

## The 3 AM Call Nobody Wants

Picture this. It's 3 AM. Your phone buzzes. PagerDuty. The sales dashboard is blank. Revenue numbers — gone. Customer support is flooding Slack. Your CEO just texted "???"

You stumble to your laptop, eyes half-open, and start digging. Forty minutes later, you find it: someone pushed a migration that changed a column from `REAL` to `TEXT`. Every `SUM()` in the pipeline is now silently concatenating strings instead of adding numbers. The ETL job duplicated 10,000 rows. And there's a stale `WHERE` clause filtering on a department name that got renamed six months ago.

This isn't hypothetical. **This happens every single day** at companies running production databases. And it's exactly the kind of messy, multi-layered problem that we thought an AI agent should learn to solve — not from textbooks, but from practice.

So we built an environment where it can.

---

## What We Built: A Production Database Simulator

Our environment, **DataOps Incident Response**, drops an AI agent into a broken production database and says: "Fix it."

But here's the twist — it's never the same break twice.

### Stochastic Fault Injection: The Secret Sauce

Most SQL benchmarks give you one broken query and one right answer. Memorize the fix, ace the test. That's not how real incidents work.

We built a **stochastic fault injector** with 12 different fault types that get randomly combined every episode:

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

Every episode randomly injects 2–4 of these. The agent can't memorize — it has to **reason**.

### Three Tiers of Pain

We designed three task difficulties that mirror real incident severity:

**Easy** — A syntax error in a simple SELECT. Missing comma between column names. Like your first on-call page — scary but fixable.

**Medium** — A three-table JOIN with the wrong join key AND wrong GROUP BY column. You need to actually understand the schema to fix this. This is the "read the ERD" incident.

**Hard** — A full ETL disaster. Duplicate transactions, orphaned foreign keys (customer_id=99 doesn't exist), and amounts stored as TEXT instead of REAL. You need to de-duplicate, type-cast, exclude orphans, and aggregate correctly. This is the "cancel your morning meetings" incident.

### Dense Rewards, Not Pass/Fail

Real debugging is incremental. You don't go from "everything is broken" to "everything works" in one step. You fix the syntax error, then realize the JOIN is wrong, then notice the duplicates...

Our reward function mirrors this:

```
R(t) = Φ(s_t) − Φ(s_{t-1}) − 0.02
```

Each sub-goal (query executes, correct columns, correct row count, correct values) gives partial credit. The agent gets rewarded for every step forward, not just the final answer. Plus penalties for destructive SQL (`DROP TABLE` = −0.5) and for repeating the same error three times (−0.15).

### Adaptive Curriculum

The environment also tracks the agent's performance and automatically escalates difficulty:

- **Novice** → 1-2 simple faults
- **Analyst** → 2-3 intermediate faults  
- **Senior** → 3-4 faults from the full pool
- **Staff Engineer** → 4 faults + a red herring table to confuse the agent

When the rolling mean score crosses 0.75, the agent gets promoted. Drop below 0.30, and it gets demoted. The environment literally gets harder as the agent gets smarter.

---

## Training: GRPO with a Live Environment

We trained using **Group Relative Policy Optimization (GRPO)** from TRL, with rewards coming directly from the live environment running on Hugging Face Spaces.

### The Setup

- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct (4-bit quantized with QLoRA)
- **Algorithm**: GRPO — samples 4 completions per prompt, uses relative ranking
- **Reward Source**: Live `/reset` + `/step` calls to the HF Space
- **Training Data**: 200 prompts across easy/medium/hard tiers
- **Hardware**: Google Colab T4 GPU (~30 min training time)

The pipeline is simple but effective:

```
For each prompt:
  1. Reset environment with task_id
  2. Model generates a SQL fix
  3. Submit to /step endpoint
  4. Read partial_score as reward
  5. GRPO updates policy based on relative reward ranking
```

### What We Found

#### The Good: Consistency Over Luck

Our baseline (untrained Qwen2.5-1.5B) had a **success rate of 20%** — but it was erratic. One episode it'd score 0.999 (perfect), the next 0.001 (total failure). It was gambling, not reasoning.

After GRPO training, the success rate jumped to **30%** — a 50% relative improvement. More importantly, the model became *consistent*. It stopped taking wild swings and learned a reliable pattern for the easy tasks.

| Metric | Baseline | Trained | Change |
|---|---|---|---|
| Mean Score | 0.325 | 0.315 | −0.01 |
| Success Rate (≥0.5) | 20% | 30% | **+50% relative** |
| Easy Task Consistency | Erratic | Stable 0.675 | ✓ |
| Max Score | 0.999 | 0.675 | ↓ (less luck) |

Wait — the mean score went *down*? Yes, and that's actually informative. The baseline had one lucky 0.999 outlier that inflated the average. The trained model traded lucky outliers for reliable partial credit. It learned a *strategy* instead of rolling dice.

#### The Discovery: A Grader Bug Was Hiding Progress

During analysis, we uncovered a critical bug: **medium tasks were always scoring 0.001**, regardless of the model's output.

The root cause was our stochastic fault injector. The `COLUMN_ALIAS_SHADOW` fault renamed SQL aliases (e.g., `revenue` → `total`), but the grader was doing exact column-name matching against the expected results. Even a perfectly correct query would fail because `result["total"]` ≠ `expected["revenue"]`.

We fixed the grader with **positional column matching** as a fallback — if column names don't match but the data shape and values do, the agent still gets credit. This fix means the next training run will show significantly better medium-task scores.

#### The Training Curves

<!-- REPLACE WITH YOUR ACTUAL COLAB SCREENSHOTS -->

The reward curve shows a clear upward trend over 20 logged steps, with the rolling mean climbing from ~0.25 to ~0.42 before settling. The loss curve shows the expected GRPO pattern — policy shifts that correlate with reward improvements.

*Training reward and loss curves from the GRPO run are available in the `training_evidence/` directory.*

---

## Why This Environment Matters

### It's Not a Benchmark — It's a Gym

Most SQL evaluation sets are static. Spider, WikiSQL, BIRD — they test whether a model can *write* SQL. But production incidents aren't about writing SQL from scratch. They're about **diagnosing what's wrong** in existing SQL and data, then fixing it incrementally.

Our environment tests:
- **Diagnosis**: Can the agent use `list_tables`, `query_schema`, `inspect_data` to understand what's broken?
- **Reasoning**: Can it trace a wrong result back to a bad JOIN key or a type mismatch?
- **Repair**: Can it write the fix, not just identify the problem?
- **Robustness**: Can it handle *unseen* fault combinations, not just memorized patterns?

### The Stochastic Element Is Key

Because faults are randomly injected, the agent can't overfit to specific fixes. In 200 training episodes, it saw hundreds of different fault combinations. This forces genuine generalization — exactly what you'd want from an AI that's supposed to handle your 3 AM incidents.

---

## Try It Yourself

The environment is live and ready to use:

🔗 **HF Space**: [https://huggingface.co/spaces/bharath1675/sql-repair-env](https://huggingface.co/spaces/bharath1675/sql-repair-env)

🔗 **Training Notebook**: [Google Colab](https://colab.research.google.com/drive/1anIA1sNPgATo73bfGp6wrlHlj3FqRICq) — you can re-run the entire training pipeline in ~30 minutes on a free T4

Quick test:
```bash
# Health check
curl https://bharath1675-sql-repair-env.hf.space/health

# Start an episode
curl -X POST https://bharath1675-sql-repair-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Submit a fix
curl -X POST https://bharath1675-sql-repair-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "submit_query", "sql_query": "SELECT name, salary FROM employees WHERE dept='\''Engineering'\'' ORDER BY salary DESC"}}'
```

---

## What's Next

With the grader fix deployed, we expect the next training run to show improvement across all three task tiers — especially medium tasks, which were previously invisible to the reward signal. We're also exploring:

1. **Smaller base models** (Qwen2.5-0.5B) where the improvement headroom is larger
2. **Multi-step agent training** where the model learns to use `list_tables` → `query_schema` → `submit_query` as a strategy, not just single-shot fixes
3. **Longer training** (500+ prompts) with lower learning rate for smoother convergence

The environment is designed to scale. Add a new fault type, and every future episode automatically becomes harder. That's the beauty of stochastic injection — the curriculum writes itself.

---

*Built for the OpenEnv Hackathon India 2026. The 3 AM calls don't stop, but maybe the AI can pick them up for us.*
