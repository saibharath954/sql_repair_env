"""
GRPO training script for the SQL Repair OpenEnv.

Trains Qwen/Qwen2.5-1.5B-Instruct (4-bit + QLoRA) using `trl.GRPOTrainer` against
a *live* SQL-repair environment exposed over HTTP (FastAPI on HF Spaces or
localhost:7860). The reward function is the environment's own `partial_score`.

Usage
-----
    BASE_URL=https://bharath1675-sql-repair-env.hf.space \
    HF_TOKEN=hf_xxx \
    python train_grpo.py

Outputs (in CWD):
    grpo_output/                      — checkpoints + final model
    training_reward_curve.png         — per-step reward
    training_loss_curve.png           — policy loss
    before_after_comparison.png       — bar chart untrained vs trained
    training_results.json             — summary metrics
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL = os.environ.get(
    "BASE_URL", "https://bharath1675-sql-repair-env.hf.space"
).rstrip("/")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = os.environ.get("HF_USERNAME", "bharath1675")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
HF_REPO_ID = os.environ.get("HF_REPO_ID", f"{HF_USERNAME}/sql-repair-grpo-qwen")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./grpo_output")

NUM_TRAIN_PROMPTS = int(os.environ.get("NUM_TRAIN_PROMPTS", "200"))
NUM_EVAL_EPISODES = int(os.environ.get("NUM_EVAL_EPISODES", "10"))
TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = (
    "You are an expert SQL engineer fixing production database incidents.\n"
    "You will be given a broken SQL query and information about the database schema.\n"
    "Your job: output ONLY the corrected SQL query.\n"
    "Rules:\n"
    "- Output raw SQL only. No markdown code blocks. No explanation.\n"
    "- Fix ALL issues: syntax errors, wrong joins, missing DISTINCT, type casts, sort order.\n"
    "- The fixed query should return the exact correct result set."
)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers — wrap the live OpenEnv server
# ─────────────────────────────────────────────────────────────────────────────
def _post(path: str, payload: dict, timeout: int = 15) -> dict:
    resp = requests.post(f"{BASE_URL}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def env_reset(task_id: str = "easy") -> dict:
    """POST /reset → observation dict."""
    data = _post("/reset", {"task_id": task_id})
    return data.get("observation", data)


def env_submit(sql: str) -> float:
    """POST /step with action_type=submit_query → partial_score."""
    data = _post(
        "/step",
        {"action_type": "submit_query", "sql_query": sql},
    )
    obs = data.get("observation", {})
    return float(obs.get("partial_score", 0.0))


def build_user_prompt(obs: dict) -> str:
    """Render the user-side text shown to the model."""
    return (
        f"TASK: {obs.get('task_description', '')}\n\n"
        f"BROKEN QUERY:\n{obs.get('broken_query', '')}\n\n"
        "Fix this query. Output only the corrected SQL:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation — call /reset NUM_TRAIN_PROMPTS times
# ─────────────────────────────────────────────────────────────────────────────
def generate_training_prompts() -> List[Dict]:
    """Generate training prompts by calling /reset on the live environment."""
    prompts: List[Dict] = []
    task_cycle = TASKS * (NUM_TRAIN_PROMPTS // len(TASKS) + 1)
    for i, task_id in enumerate(task_cycle[:NUM_TRAIN_PROMPTS]):
        try:
            obs = env_reset(task_id)
            user_prompt = build_user_prompt(obs)
            prompts.append(
                {
                    "prompt": user_prompt,
                    "task_id": task_id,
                    "broken_query": obs.get("broken_query", ""),
                    "task_description": obs.get("task_description", ""),
                }
            )
        except Exception as e:
            print(f"[warn] failed to fetch prompt {i} ({task_id}): {e}")
    print(f"Generated {len(prompts)} training prompts")
    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# Reward function — calls live env /reset + /step and returns partial_score
# ─────────────────────────────────────────────────────────────────────────────
def make_reward_fn(prompts_meta: List[Dict]):
    """
    Build a reward_fn that maps completion → partial_score by calling the live env.
    Uses ThreadPoolExecutor to evaluate up to 4 completions in parallel.
    """
    prompt_to_task = {p["prompt"]: p["task_id"] for p in prompts_meta}
    episode_counter = {"n": 0}

    def _evaluate_one(completion: str, prompt: str) -> float:
        task_id = prompt_to_task.get(prompt, "easy")
        sql = (completion or "").strip()
        if not sql:
            return 0.0
        # strip markdown fences if the model added them anyway
        if sql.startswith("```"):
            sql = sql.strip("`")
            if sql.lower().startswith("sql"):
                sql = sql[3:]
            sql = sql.strip()
        try:
            requests.post(
                f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=15
            ).raise_for_status()
            return env_submit(sql)
        except Exception:
            return 0.0

    def reward_fn(
        completions: List, prompts: List[str] = None, **kwargs
    ) -> List[float]:
        # `completions` may be list[str] or list[list[{role,content}]] depending on TRL version
        if completions and isinstance(completions[0], list):
            flat = ["".join(turn.get("content", "") for turn in c) for c in completions]
        else:
            flat = [str(c) for c in completions]

        # If prompts are conversational (list of message dicts), reduce to user text
        flat_prompts: List[str] = []
        if prompts is None:
            flat_prompts = [""] * len(flat)
        else:
            for p in prompts:
                if isinstance(p, list):
                    flat_prompts.append(
                        next(
                            (m.get("content", "") for m in p if m.get("role") == "user"),
                            "",
                        )
                    )
                else:
                    flat_prompts.append(str(p))

        with ThreadPoolExecutor(max_workers=4) as pool:
            rewards = list(pool.map(_evaluate_one, flat, flat_prompts))

        for r in rewards:
            episode_counter["n"] += 1
            print(f"Episode {episode_counter['n']}: reward={r:.3f}")
        return rewards

    return reward_fn


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — greedy decode against the live env
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, tokenizer, n_episodes: int = 10) -> Tuple[float, List[float]]:
    """Run n episodes with the given model and return (mean, per-episode list)."""
    import torch

    scores: List[float] = []
    model.eval()
    for i in range(n_episodes):
        task_id = TASKS[i % len(TASKS)]
        try:
            obs = env_reset(task_id)
            user_prompt = build_user_prompt(obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            if completion.startswith("```"):
                completion = completion.strip("`")
                if completion.lower().startswith("sql"):
                    completion = completion[3:]
                completion = completion.strip()
            score = env_submit(completion) if completion else 0.0
            scores.append(float(score))
            print(f"  Eval {i + 1}/{n_episodes} [{task_id}]: score={score:.3f}")
        except Exception as e:
            print(f"  Eval {i + 1}/{n_episodes} failed: {e}")
            scores.append(0.0)

    mean = sum(scores) / len(scores) if scores else 0.0
    return mean, scores


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — reward curve, loss curve, before/after bars
# ─────────────────────────────────────────────────────────────────────────────
def save_plots(
    train_rewards: List[float],
    train_losses: List[float],
    baseline_score: float,
    trained_score: float,
) -> None:
    if train_rewards:
        steps = list(range(len(train_rewards)))
        plt.figure(figsize=(10, 5))
        plt.plot(steps, train_rewards, "b-", linewidth=1.5, alpha=0.5, label="Per-step reward")
        window = max(1, len(train_rewards) // 20)
        smoothed = [
            sum(train_rewards[max(0, i - window): i + 1])
            / max(1, min(i + 1, window + 1))
            for i in range(len(train_rewards))
        ]
        plt.plot(steps, smoothed, "r-", linewidth=2.5, label=f"Rolling mean (window={window})")
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Episode Reward (partial_score)", fontsize=12)
        plt.title(
            "Training Reward Curve — DataOps Incident Response Env\nQwen2.5-1.5B + GRPO",
            fontsize=13,
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_reward_curve.png", dpi=150)
        plt.close()
        print("Saved: training_reward_curve.png")

    if train_losses:
        loss_steps = list(range(len(train_losses)))
        plt.figure(figsize=(10, 5))
        plt.plot(loss_steps, train_losses, "g-", linewidth=1.5)
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Policy Loss", fontsize=12)
        plt.title(
            "Training Loss Curve — DataOps Incident Response Env\nQwen2.5-1.5B + GRPO",
            fontsize=13,
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_loss_curve.png", dpi=150)
        plt.close()
        print("Saved: training_loss_curve.png")

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Untrained\n(Qwen2.5-1.5B)", f"Trained\n(GRPO, {NUM_TRAIN_PROMPTS} eps)"],
        [baseline_score, trained_score],
        color=["#e74c3c", "#2ecc71"],
        width=0.5,
        edgecolor="black",
        linewidth=1.2,
    )
    for bar, score in zip(bars, [baseline_score, trained_score]):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    plt.ylim(0, max(1.1, max(baseline_score, trained_score) * 1.2))
    plt.ylabel("Mean Episode Score (partial_score)", fontsize=12)
    plt.title(
        "Before vs After Training\nDataOps Incident Response Env — Qwen2.5-1.5B + GRPO",
        fontsize=12,
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("before_after_comparison.png", dpi=150)
    plt.close()
    print("Saved: before_after_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import GRPOConfig, GRPOTrainer

    print(f"Environment: {BASE_URL}")
    print(f"Model:       {MODEL_NAME}")

    # ── Health check ──
    health = requests.get(f"{BASE_URL}/health", timeout=15)
    health.raise_for_status()
    print(f"Environment healthy: {health.json()}")

    # ── Generate dataset ──
    print("\n=== Generating training prompts ===")
    prompts_meta = generate_training_prompts()
    if not prompts_meta:
        print("ERROR: no prompts generated; aborting.")
        sys.exit(1)

    dataset = Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p["prompt"]},
                ]
                for p in prompts_meta
            ],
            "task_id": [p["task_id"] for p in prompts_meta],
        }
    )

    # ── Tokenizer + 4-bit base model ──
    print(f"\n=== Loading {MODEL_NAME} (4-bit NF4 + QLoRA) ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for generation in GRPO

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── PEFT/LoRA config — required to train a 4-bit base model ──
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # ── Baseline evaluation (untrained) ──
    print("\n=== Baseline evaluation (untrained model) ===")
    baseline_mean, baseline_scores = evaluate_model(
        model, tokenizer, n_episodes=NUM_EVAL_EPISODES
    )
    print(f"Baseline mean score: {baseline_mean:.4f}")

    # ── GRPO config ──
    grpo_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        max_prompt_length=512,
        max_completion_length=256,
        num_generations=4,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        use_vllm=False,
    )

    reward_fn = make_reward_fn(prompts_meta)

    # ── Metric capture callback ──
    train_rewards: List[float] = []
    train_losses: List[float] = []

    class MetricCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            for key in ("reward", "rewards/reward_fn", "train/reward"):
                if key in logs:
                    train_rewards.append(float(logs[key]))
                    break
            if "loss" in logs:
                train_losses.append(float(logs["loss"]))

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MetricCallback()],
    )

    print("\n=== Training ===")
    t0 = time.time()
    trainer.train()
    print(f"Training finished in {(time.time() - t0) / 60:.1f} min")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    # ── Post-training evaluation ──
    print("\n=== Post-training evaluation ===")
    trained_mean, trained_scores = evaluate_model(
        trainer.model, tokenizer, n_episodes=NUM_EVAL_EPISODES
    )
    print(f"Trained mean score: {trained_mean:.4f}")

    # ── Results table ──
    delta = trained_mean - baseline_mean
    base_succ = sum(s >= 0.5 for s in baseline_scores) / max(1, len(baseline_scores))
    train_succ = sum(s >= 0.5 for s in trained_scores) / max(1, len(trained_scores))
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<32} {'Baseline':>10} {'Trained':>10} {'Delta':>10}")
    print("-" * 60)
    print(f"{'Mean episode score':<32} {baseline_mean:>10.4f} {trained_mean:>10.4f} {delta:>+10.4f}")
    print(f"{'Success rate (score>=0.5)':<32} {base_succ:>10.2%} {train_succ:>10.2%} {train_succ - base_succ:>+10.2%}")
    print("=" * 60)

    # ── Plots ──
    print("\n=== Saving plots ===")
    save_plots(train_rewards, train_losses, baseline_mean, trained_mean)

    # ── Persist results ──
    results = {
        "model": MODEL_NAME,
        "base_url": BASE_URL,
        "baseline_score": baseline_mean,
        "trained_score": trained_mean,
        "improvement": delta,
        "baseline_success_rate": base_succ,
        "trained_success_rate": train_succ,
        "baseline_scores": baseline_scores,
        "trained_scores": trained_scores,
        "train_logged_steps": len(train_rewards),
        "num_train_prompts": NUM_TRAIN_PROMPTS,
    }
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to training_results.json")

    # ── Push to HF Hub ──
    if HF_TOKEN:
        print(f"\n=== Pushing LoRA adapters to HF Hub: {HF_REPO_ID} ===")
        try:
            from huggingface_hub import HfApi, create_repo

            create_repo(HF_REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")
            HfApi(token=HF_TOKEN).upload_folder(
                folder_path=OUTPUT_DIR,
                repo_id=HF_REPO_ID,
                repo_type="model",
                commit_message="GRPO-tuned QLoRA adapters for SQL repair env",
            )
            print(f"Pushed to https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            print(f"[warn] HF Hub push failed: {e}")

    print(
        "\nDone! Artifacts:\n"
        "  • training_reward_curve.png\n"
        "  • training_loss_curve.png\n"
        "  • before_after_comparison.png\n"
        "  • training_results.json\n"
        f"  • {OUTPUT_DIR}/ (model checkpoints + adapters)"
    )


if __name__ == "__main__":
    main()
