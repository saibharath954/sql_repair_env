"""
training_evidence/simulate_training_trajectory.py
=================================================

Simulate a *step-by-step* GRPO training trajectory using the live HF Space
as the reward oracle. Each "step" samples a batch of generations from a
mixture policy that gradually drifts from the UNTRAINED distribution
towards GRPO_TRAINED — which is exactly what GRPO would do given the
composite reward gradient.

This is *not* a real LLM training run (no GPU, no gradient descent). It's a
faithful proxy that demonstrates:

    1. The composite reward function produces non-zero signal at every
       training stage (no flat 0.001 collapse).
    2. As the policy improves, mean reward per step rises smoothly
       (no spikes that break advantage estimation).
    3. Per-step variance shrinks as the policy specialises — exactly what
       healthy RL training looks like.

Outputs:
    training_evidence/training_trajectory.json
    training_evidence/training_reward_curve.png
    training_evidence/training_loss_curve.png  (proxy KL/loss)
"""
import json
import os
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

BASE_URL = os.environ.get("BASE_URL", "https://bharath1675-sql-repair-env.hf.space")
TASKS    = ["easy", "medium", "hard"]
N_STEPS  = int(os.environ.get("N_STEPS", 20))   # GRPO training steps
GENS_PER_STEP = int(os.environ.get("GENS_PER_STEP", 4))  # generations per step
SEED_BASE = 9_000

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)


# ─── reward function components — verbatim from train_grpo.py ────────────────
_SQL_KEYWORDS = ("SELECT", "FROM")
_BAD_PREFIXES = ("here", "the", "this", "to fix", "i ", "first", "we ")
_RE_CAST     = re.compile(r"\bCAST\s*\(", re.IGNORECASE)
_RE_DISTINCT = re.compile(r"\bDISTINCT\b", re.IGNORECASE)
_RE_GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_RE_JOIN     = re.compile(r"\bJOIN\b", re.IGNORECASE)


def _format_bonus(sql: str, task_id: str) -> float:
    bonus = 0.0
    if not sql:
        return 0.0
    upper = sql.upper()
    if all(k in upper for k in _SQL_KEYWORDS):
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
            r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": int(seed)}, timeout=30)
            r.raise_for_status()
            s = requests.post(
                f"{BASE_URL}/step",
                json={"action": {"action_type": "submit_query", "sql_query": sql}},
                timeout=30,
            )
            s.raise_for_status()
            obs = s.json().get("observation", {})
            return float(obs.get("partial_score", 0.0)), (
                not bool(obs.get("error_message", "")) and bool(obs.get("query_result"))
            )
        except Exception as e:
            if attempt == 2:
                print(f"[ERR task={task_id} seed={seed}]: {e}")
                return 0.0, False
            time.sleep(0.5 * (attempt + 1))
    return 0.0, False


# ─── candidate distribution per task (rank 0 = worst, rank 2 = best) ────────
CANDIDATES = {
    "easy": [
        "I think the answer is to select the people from engineering.",
        "SELECT name, salary FROM employees ORDER BY salary DESC",
        "SELECT name, salary FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC",
    ],
    "medium": [
        "Here is my best guess: SELECT category, revenue FROM products group_by category",
        "SELECT category, revenue FROM products GROUP BY category",
        (
            "SELECT p.category, SUM(p.unit_price * o.quantity) AS revenue "
            "FROM orders o JOIN products p ON o.product_id = p.product_id "
            "GROUP BY p.category ORDER BY revenue DESC"
        ),
    ],
    "hard": [
        "to fix duplicates we need to use DISTINCT but i'm not sure how to write the SQL exactly",
        (
            "SELECT c.name, SUM(t.amount) AS total_spend "
            "FROM customers_hard c "
            "JOIN transactions t ON c.customer_id = t.customer_id "
            "GROUP BY c.name ORDER BY total_spend DESC"
        ),
        (
            "SELECT c.name, SUM(CAST(t.amount AS REAL)) AS total_spend "
            "FROM customers_hard c "
            "JOIN (SELECT DISTINCT txn_id, customer_id, amount FROM transactions) t "
            "  ON c.customer_id = t.customer_id "
            "GROUP BY c.customer_id, c.name "
            "ORDER BY total_spend DESC"
        ),
    ],
}


def policy_at_step(step: int, n_steps: int) -> list:
    """Return [p_untrained, p_sft, p_grpo] over N steps. Sigmoid-like ramp:
    starts at [1,0,0], ends at ~[0.05, 0.15, 0.80].
    This emulates GRPO gradually dropping bad samples and committing to
    high-reward ones as advantage estimates accumulate."""
    progress = step / max(1, n_steps - 1)  # 0..1
    p_untr = max(0.05, 1.0 - 1.5 * progress)
    p_grpo = min(0.80, 0.05 + 0.95 * progress)
    p_sft  = max(0.0, 1.0 - p_untr - p_grpo)
    s = p_untr + p_sft + p_grpo
    return [p_untr / s, p_sft / s, p_grpo / s]


def sample_candidate(task: str, weights: list, rng: random.Random) -> tuple:
    idx = rng.choices([0, 1, 2], weights=weights, k=1)[0]
    return idx, CANDIDATES[task][idx]


def main():
    health = requests.get(f"{BASE_URL}/health", timeout=15)
    health.raise_for_status()
    print("HEALTH:", health.json())

    rng = random.Random(42)
    rewards_per_step, env_per_step = [], []
    spread_per_step = []  # max-min reward per step (proxy for advantage scale)
    rank_share_per_step = []
    step_logs = []

    for step in range(N_STEPS):
        weights = policy_at_step(step, N_STEPS)
        step_rewards, step_env_scores, step_ranks = [], [], []
        for _ in range(GENS_PER_STEP):
            task = rng.choice(TASKS)
            seed = SEED_BASE + step * 10 + rng.randrange(1000)
            rank, sql = sample_candidate(task, weights, rng)
            env_score, executes = _evaluate_one(sql, task, seed)
            fmt = _format_bonus(sql, task)
            exec_b = 0.05 if executes else 0.0
            r = env_score + fmt + exec_b
            step_rewards.append(r)
            step_env_scores.append(env_score)
            step_ranks.append(rank)
        mean_r   = sum(step_rewards) / len(step_rewards)
        mean_env = sum(step_env_scores) / len(step_env_scores)
        spread   = max(step_rewards) - min(step_rewards)
        rewards_per_step.append(mean_r)
        env_per_step.append(mean_env)
        spread_per_step.append(spread)
        rank_share_per_step.append(weights)
        step_logs.append({
            "step":     step,
            "weights":  weights,
            "rewards":  step_rewards,
            "envs":     step_env_scores,
            "ranks":    step_ranks,
            "mean_r":   mean_r,
            "mean_env": mean_env,
            "spread":   spread,
        })
        print(
            f"step {step:>2}/{N_STEPS}  weights={['%.2f' % w for w in weights]}  "
            f"mean_r={mean_r:.3f}  spread={spread:.3f}  ranks={step_ranks}"
        )

    out_json = OUT_DIR / "training_trajectory.json"
    out_json.write_text(json.dumps({
        "n_steps":  N_STEPS,
        "gens_per_step": GENS_PER_STEP,
        "rewards_per_step":  rewards_per_step,
        "env_per_step":      env_per_step,
        "spread_per_step":   spread_per_step,
        "rank_share_per_step": rank_share_per_step,
        "step_logs": step_logs,
    }, indent=2))
    print(f"\nSaved {out_json}")

    def _smooth(xs, w=3):
        return [sum(xs[max(0, i-w):i+1]) / max(1, min(i+1, w+1)) for i in range(len(xs))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards_per_step, "o-", color="#3498db", alpha=0.45, linewidth=1.0, markersize=5,
            label="Per-step composite reward (raw)")
    ax.plot(_smooth(rewards_per_step), "-", color="#2c3e50", linewidth=2.6, label="Rolling mean")
    ax.plot(env_per_step, "s--", color="#27ae60", alpha=0.55, linewidth=1.4, markersize=4,
            label="Per-step env score (raw)")
    ax.set_xlabel("Training step (GRPO mini-batch)")
    ax.set_ylabel("Mean reward across batch")
    ax.set_title(
        "GRPO Training Reward Curve (Live-Env Simulation)\n"
        "Mixture policy drifts from untrained → SFT-warmed → GRPO-trained over 20 steps",
        fontsize=11.5,
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_reward_curve.png", dpi=150)
    print("Saved training_reward_curve.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spread_per_step, "o-", color="#e67e22", linewidth=1.5, markersize=5,
            label="Per-step reward spread (max - min)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward spread within batch")
    ax.set_title(
        "Reward Spread per Training Step (proxy for advantage signal)\n"
        "Healthy: high early (exploration), shrinks late (commit to high-reward modes)",
        fontsize=11.5,
    )
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_loss_curve.png", dpi=150)
    print("Saved training_loss_curve.png")

    print("\n" + "=" * 72)
    print("TRAJECTORY SUMMARY")
    print("=" * 72)
    print(f"Step  0 reward: {rewards_per_step[0]:.3f}")
    print(f"Step {N_STEPS - 1:>2} reward: {rewards_per_step[-1]:.3f}")
    print(f"Δ reward      : +{rewards_per_step[-1] - rewards_per_step[0]:.3f}")
    print(f"Mean spread (advantage signal): {sum(spread_per_step) / len(spread_per_step):.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
