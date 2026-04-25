"""
Smoke test for the new reward signal.

Verifies that:
1. The grader gives a *real* zero (not 0.001) for completely failed queries.
2. A correct single-shot submit_query reaches near-1.0 on ALL three tasks.
3. The gold answers from action_schema score >= 0.9 even with fault injection
   when run against the canonical broken_query (no random extra faults).
4. The composite reward (env + format + exec) produces a meaningful spread
   across 4 candidate completions per task — exactly the spread GRPO needs.

Run with:
    cd sql_repair_env
    source venv/bin/activate
    python test_reward_signal.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
from server.environment import SQLRepairEnvironment
from server.grader import compute_score, SUBGOALS
from server.tasks import TASKS
from models import SQLRepairAction


# ─── format/exec auxiliary reward (mirrors train_grpo.py) ───────────────────
_RE_CAST     = re.compile(r"\bCAST\s*\(", re.IGNORECASE)
_RE_DISTINCT = re.compile(r"\bDISTINCT\b", re.IGNORECASE)
_RE_GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_RE_JOIN     = re.compile(r"\bJOIN\b", re.IGNORECASE)
_BAD_PREFIXES = ("here", "the", "this", "to fix", "i ", "first", "we ")


def _format_bonus(sql: str, task_id: str) -> float:
    bonus = 0.0
    if not sql:
        return 0.0
    upper = sql.upper()
    if "SELECT" in upper and "FROM" in upper:
        bonus += 0.05
    head = sql.lstrip().lower()[:30]
    if not any(head.startswith(b) for b in _BAD_PREFIXES) and not head.startswith("```"):
        bonus += 0.03
    if 20 <= len(sql) <= 600:
        bonus += 0.02
    if task_id == "medium":
        if _RE_GROUP_BY.search(sql):
            bonus += 0.04
        if _RE_JOIN.search(sql):
            bonus += 0.03
    elif task_id == "hard":
        if _RE_CAST.search(sql):
            bonus += 0.05
        if _RE_DISTINCT.search(sql) or _RE_GROUP_BY.search(sql):
            bonus += 0.03
    return min(bonus, 0.20)


def _make_no_fault_env() -> SQLRepairEnvironment:
    """Disable fault injection so we can verify the grader on the canonical
    broken_query — exactly what the test suite uses."""
    e = SQLRepairEnvironment()

    def _no_op_inject(conn, task_id, broken_query, seed=None):
        return {"injected_faults": [], "broken_query": broken_query, "fault_count": 0}

    e._fault_injector.inject = _no_op_inject  # type: ignore[assignment]
    return e


def _composite_reward(env: SQLRepairEnvironment, task_id: str, sql: str) -> dict:
    """Reset, submit, return env score + auxiliary terms + composite total."""
    env.reset(task_id=task_id)
    obs = env.step(SQLRepairAction(action_type="submit_query", sql_query=sql))
    env_score = float(obs.partial_score)
    executes = (not obs.error_message) and bool(obs.query_result)
    fmt = _format_bonus(sql, task_id)
    exec_bonus = 0.05 if executes else 0.0
    return {
        "env":   round(env_score, 4),
        "fmt":   round(fmt, 3),
        "exec":  round(exec_bonus, 3),
        "total": round(env_score + fmt + exec_bonus, 4),
        "err":   obs.error_message[:80] if obs.error_message else "",
    }


# ─── candidate completions per task — covers a "GRPO group" ─────────────────
CANDIDATES = {
    "easy": [
        # 0: garbage / off-task
        "I think the answer is here",
        # 1: broken (the original SQL)
        TASKS["easy"]["broken_query"],
        # 2: partial — wrong WHERE
        "SELECT name, salary FROM employees ORDER BY salary DESC",
        # 3: gold
        TASKS["easy"]["action_schema"]["sql_query"],
    ],
    "medium": [
        "explain this query",
        TASKS["medium"]["broken_query"],
        # half-fix — correct join but wrong group_by
        "SELECT p.category, SUM(p.unit_price * o.quantity) AS revenue "
        "FROM orders o JOIN products p ON o.product_id = p.product_id "
        "GROUP BY o.product_id ORDER BY revenue DESC",
        TASKS["medium"]["action_schema"]["sql_query"],
    ],
    "hard": [
        "the query is",
        TASKS["hard"]["broken_query"],
        # naive — no DISTINCT, no CAST
        "SELECT c.name, SUM(t.amount) AS total_spend FROM customers_hard c "
        "JOIN transactions t ON c.customer_id = t.customer_id "
        "GROUP BY c.name ORDER BY total_spend DESC",
        TASKS["hard"]["action_schema"]["sql_query"],
    ],
}


def main():
    env = _make_no_fault_env()
    print("=" * 96)
    print("REWARD SIGNAL SMOKE TEST")
    print("=" * 96)
    print(f"{'task':<8} {'cand':<5} {'env':>7} {'fmt':>6} {'exec':>6} {'total':>7}  preview")
    print("-" * 96)

    for tid in ["easy", "medium", "hard"]:
        rewards = []
        for i, cand in enumerate(CANDIDATES[tid]):
            res = _composite_reward(env, tid, cand)
            rewards.append(res["total"])
            preview = (cand[:60] + "…") if len(cand) > 60 else cand
            print(
                f"{tid:<8} {i:<5} {res['env']:>7.3f} {res['fmt']:>6.3f} {res['exec']:>6.3f} "
                f"{res['total']:>7.3f}  {preview}"
            )
        # Spread = (max - min) reward across the 4 candidates. GRPO needs > 0.
        spread = max(rewards) - min(rewards)
        std = (sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)) ** 0.5
        print(f"{tid:<8} ↳ spread={spread:.3f}  std={std:.3f}  "
              f"(GRPO needs std > 0 to learn)")
        print("-" * 96)

    print("\n=== Pass criteria ===")
    print("- Garbage candidates score 0.0 env (not 0.001 floor).")
    print("- Gold candidates score >= 0.9 env on every task.")
    print("- Each task has spread > 0.3 across the 4 candidates.")


if __name__ == "__main__":
    main()
