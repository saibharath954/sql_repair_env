"""End-to-end smoke test against the LIVE HF Space.

Verifies the seeded reward function works against the live server (still
running the OLD grader). Even with the old grader, we should see:
- Per-seed reproducibility (same seed → same env_score)
- Composite reward gives non-trivial spread across (garbage, partial, gold).

Once the SERVER changes (new grader) are deployed, the medium/hard "gold"
should jump from ~0.7 to ~1.0.
"""

import re
import time
import requests

BASE_URL = "https://bharath1675-sql-repair-env.hf.space"

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


def _evaluate_one(sql: str, task_id: str, seed: int):
    if not sql:
        return 0.0, False
    for attempt in range(3):
        try:
            r = requests.post(
                f"{BASE_URL}/reset", json={"task_id": task_id, "seed": int(seed)}, timeout=20,
            )
            r.raise_for_status()
            s = requests.post(
                f"{BASE_URL}/step",
                json={"action": {"action_type": "submit_query", "sql_query": sql}},
                timeout=20,
            )
            s.raise_for_status()
            obs = s.json().get("observation", {})
            return float(obs.get("partial_score", 0.0)), (
                not bool(obs.get("error_message"))
                and bool(obs.get("query_result"))
            )
        except Exception as e:
            if attempt == 2:
                print(f"[ERR task={task_id} seed={seed}] {e}")
                return 0.0, False
            time.sleep(0.5 * (attempt + 1))
    return 0.0, False


GOLD = {
    "easy":   "SELECT name, salary FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC",
    "medium": (
        "SELECT p.category, SUM(p.unit_price * o.quantity) AS revenue "
        "FROM orders o JOIN products p ON o.product_id = p.product_id "
        "GROUP BY p.category ORDER BY revenue DESC"
    ),
    "hard":   (
        "SELECT c.name, SUM(CAST(t.amount AS REAL)) AS total_spend "
        "FROM customers_hard c "
        "JOIN (SELECT DISTINCT txn_id, customer_id, amount FROM transactions) t "
        "  ON c.customer_id = t.customer_id "
        "GROUP BY c.customer_id, c.name "
        "ORDER BY total_spend DESC"
    ),
}
PARTIAL = {
    "easy":   "SELECT name, salary FROM employees ORDER BY salary DESC",
    "medium": "SELECT category FROM products GROUP BY category",
    "hard":   "SELECT c.name, SUM(t.amount) AS total_spend FROM customers_hard c JOIN transactions t ON c.customer_id = t.customer_id GROUP BY c.name ORDER BY total_spend DESC",
}
GARBAGE = "I think the answer is SELECT * FROM xyz"


def main():
    print(f"{'task':<8} {'seed':>5} {'label':<8} {'env':>7} {'fmt':>6} {'exec':>5} {'total':>7}")
    print("-" * 60)
    rows_by_task = {t: {} for t in GOLD}
    for task in ["easy", "medium", "hard"]:
        for seed in [9000, 9001]:
            for label, sql in [("garbage", GARBAGE), ("partial", PARTIAL[task]), ("gold", GOLD[task])]:
                env_score, executes = _evaluate_one(sql, task, seed)
                fmt = _format_bonus(sql, task)
                exec_b = 0.05 if executes else 0.0
                total = env_score + fmt + exec_b
                rows_by_task[task].setdefault(label, []).append(total)
                print(f"{task:<8} {seed:>5} {label:<8} {env_score:>7.3f} {fmt:>6.3f} {exec_b:>5.2f} {total:>7.3f}")
        print("-" * 60)

    print("\nSpread per task (max-min across {garbage, partial, gold}):")
    for task in ["easy", "medium", "hard"]:
        all_totals = sum(rows_by_task[task].values(), [])
        spread = max(all_totals) - min(all_totals)
        print(f"  {task:<8} spread={spread:.3f}")


if __name__ == "__main__":
    main()
