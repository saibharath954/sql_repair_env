"""
Programmatic builder for train_grpo.ipynb.

Run:
    python build_notebook.py        # writes train_grpo.ipynb in the same dir
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ─── Cell 0 — Title ──────────────────────────────────────────────────────────
md(
    """# DataOps Incident Response — GRPO Training Notebook

Train **Qwen2.5-1.5B-Instruct** to diagnose and fix corrupted production databases by
learning directly from a *live* OpenEnv SQL-repair environment running on a Hugging Face
Space. The reward signal is the environment's own `partial_score` returned by `/step`.

- **Algorithm:** GRPO (Group Relative Policy Optimization) via `trl.GRPOTrainer`
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4 + QLoRA adapters)
- **Environment:** Live HF Space exposing `/reset` and `/step`
- **Estimated runtime on T4 GPU:** ~30-45 minutes for 200 prompts × 2 epochs

Pipeline: connect → generate prompts → baseline eval → train → post-eval → plots → push."""
)


# ─── Cell 1 — Install ────────────────────────────────────────────────────────
md("## Cell 1 — Install Dependencies\nInstall TRL, Transformers, PEFT and friends. Skip if already installed.")
code(
    """
%%capture
!pip install -q "trl>=0.12" "transformers>=4.45" "datasets>=2.20" \\
    "bitsandbytes>=0.43" "accelerate>=0.34" "peft>=0.13" \\
    matplotlib requests "huggingface_hub>=0.25"
"""
)


# ─── Cell 2 — Configuration ──────────────────────────────────────────────────
md("## Cell 2 — Configuration\nReplace `BASE_URL` with your HF Space URL. Optionally set `HF_TOKEN` to push the trained model.")
code(
    """
import os

BASE_URL          = "https://bharath1675-sql-repair-env.hf.space"   # ← your HF Space URL
MODEL_NAME        = "Qwen/Qwen2.5-1.5B-Instruct"
HF_TOKEN          = ""        # ← optional: HF token for push_to_hub
HF_USERNAME       = "bharath1675"
HF_REPO_ID        = f"{HF_USERNAME}/sql-repair-grpo-qwen"
OUTPUT_DIR        = "./grpo_output"
NUM_TRAIN_PROMPTS = 200
NUM_EVAL_EPISODES = 10
TASKS             = ["easy", "medium", "hard"]

SYSTEM_PROMPT = (
    "You are an expert SQL engineer fixing production database incidents.\\n"
    "You will be given a broken SQL query and information about the database schema.\\n"
    "Your job: output ONLY the corrected SQL query.\\n"
    "Rules:\\n"
    "- Output raw SQL only. No markdown code blocks. No explanation.\\n"
    "- Fix ALL issues: syntax errors, wrong joins, missing DISTINCT, type casts, sort order.\\n"
    "- The fixed query should return the exact correct result set."
)
print(f"Targeting environment: {BASE_URL}")
"""
)


# ─── Cell 3 — Verify environment ─────────────────────────────────────────────
md(
    """## Step 1 — Verify Environment Connection

Before training, sanity-check that the live HF Space is reachable and that
`/reset` + `/step` round-trip correctly. Any failure here means the URL or
the deployment is wrong — fix it before continuing."""
)
code(
    """
import requests

try:
    health = requests.get(f"{BASE_URL}/health", timeout=15)
    health.raise_for_status()
    print("HEALTH:", health.json())

    reset = requests.post(f"{BASE_URL}/reset", json={"task_id": "easy"}, timeout=15)
    reset.raise_for_status()
    obs = reset.json()["observation"]
    print("\\nTask description:", obs.get("task_description", "")[:120])
    print("Broken query    :", obs.get("broken_query", "")[:120])

    step = requests.post(
        f"{BASE_URL}/step",
        json={"action_type": "submit_query",
              "sql_query": "SELECT name, salary FROM employees WHERE dept='Engineering' ORDER BY salary DESC"},
        timeout=15,
    )
    step.raise_for_status()
    print("\\nStep partial_score:", step.json()["observation"]["partial_score"])
except Exception as e:
    print("Environment check failed:", e)
    raise
"""
)


# ─── Cell 4 — Dataset ────────────────────────────────────────────────────────
md(
    """## Step 2 — Generate Training Dataset

We collect 200 prompts by calling `/reset` 200 times across the easy / medium / hard
task tiers. Each prompt embeds the task description and the broken SQL query the
model must repair."""
)
code(
    """
def build_user_prompt(obs: dict) -> str:
    return (
        f"TASK: {obs.get('task_description', '')}\\n\\n"
        f"BROKEN QUERY:\\n{obs.get('broken_query', '')}\\n\\n"
        "Fix this query. Output only the corrected SQL:"
    )


def generate_training_prompts():
    prompts = []
    cycle = TASKS * (NUM_TRAIN_PROMPTS // len(TASKS) + 1)
    for i, task_id in enumerate(cycle[:NUM_TRAIN_PROMPTS]):
        try:
            resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=15)
            resp.raise_for_status()
            obs = resp.json().get("observation", {})
            prompts.append({
                "prompt": build_user_prompt(obs),
                "task_id": task_id,
                "broken_query": obs.get("broken_query", ""),
            })
        except Exception as e:
            print(f"[warn] failed prompt {i} ({task_id}): {e}")
    print(f"Generated {len(prompts)} training prompts")
    return prompts


prompts_meta = generate_training_prompts()
print("\\n--- Sample prompt ---")
print(prompts_meta[0]["prompt"])
"""
)


# ─── Cell 5 — Reward function ────────────────────────────────────────────────
md(
    """## Step 3 — Define the Reward Function

`reward_fn` is the bridge between the policy and the environment. For every model
completion we:
1. `POST /reset` with the matching task_id (clean DB).
2. `POST /step` with `action_type="submit_query"` and the SQL the model produced.
3. Read `partial_score` (∈ [0, 1]) and use it as the scalar reward.

Up to 4 completions are evaluated in parallel via `ThreadPoolExecutor` to keep the
GPU fed during rollouts."""
)
code(
    """
from concurrent.futures import ThreadPoolExecutor


def make_reward_fn(prompts_meta):
    prompt_to_task = {p["prompt"]: p["task_id"] for p in prompts_meta}
    counter = {"n": 0}

    def _evaluate_one(completion: str, prompt: str) -> float:
        task_id = prompt_to_task.get(prompt, "easy")
        sql = (completion or "").strip()
        if not sql:
            return 0.0
        if sql.startswith("```"):
            sql = sql.strip("`")
            if sql.lower().startswith("sql"):
                sql = sql[3:]
            sql = sql.strip()
        try:
            r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=15)
            r.raise_for_status()
            s = requests.post(
                f"{BASE_URL}/step",
                json={"action_type": "submit_query", "sql_query": sql},
                timeout=15,
            )
            s.raise_for_status()
            return float(s.json().get("observation", {}).get("partial_score", 0.0))
        except Exception:
            return 0.0

    def reward_fn(completions, prompts=None, **kwargs):
        if completions and isinstance(completions[0], list):
            flat = ["".join(t.get("content", "") for t in c) for c in completions]
        else:
            flat = [str(c) for c in completions]
        flat_prompts = []
        for p in prompts or [""] * len(flat):
            if isinstance(p, list):
                flat_prompts.append(
                    next((m.get("content", "") for m in p if m.get("role") == "user"), "")
                )
            else:
                flat_prompts.append(str(p))
        with ThreadPoolExecutor(max_workers=4) as pool:
            rewards = list(pool.map(_evaluate_one, flat, flat_prompts))
        for r in rewards:
            counter["n"] += 1
            print(f"Episode {counter['n']}: reward={r:.3f}")
        return rewards

    return reward_fn


reward_fn = make_reward_fn(prompts_meta)

# Quick sanity test: send a deliberately broken SQL string and confirm we get a float back.
sample = prompts_meta[0]["prompt"]
test_reward = reward_fn(["SELECT * FROM employees"], prompts=[sample])
print(f"\\nSanity reward: {test_reward}")
"""
)


# ─── Cell 6 — Load model ─────────────────────────────────────────────────────
md(
    """## Step 4 — Load Qwen2.5-1.5B (4-bit Quantized)

We load the base model with NF4 4-bit quantization (bitsandbytes) and stage QLoRA
adapters for the attention projections. This drops the GPU memory footprint to
fit comfortably on a T4 (~16 GB)."""
)
code(
    """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

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

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded {MODEL_NAME}  |  parameters: {n_params/1e6:.1f}M")
print(f"Memory footprint   : {torch.cuda.memory_allocated()/1024**3:.2f} GiB" if torch.cuda.is_available() else "CPU only")
"""
)


# ─── Cell 7 — Baseline eval ──────────────────────────────────────────────────
md(
    """## Step 5 — Baseline Evaluation (Before Training)

Greedy-decode the *untrained* model on 10 environment episodes (mix of easy/medium/hard)
to establish a baseline `partial_score`. Anything we gain after GRPO training is
*on top of* this number."""
)
code(
    """
def evaluate_model(model, tokenizer, n_episodes=10):
    model.eval()
    scores = []
    for i in range(n_episodes):
        task_id = TASKS[i % len(TASKS)]
        try:
            obs = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=15).json()["observation"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs)},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            comp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if comp.startswith("```"):
                comp = comp.strip("`")
                if comp.lower().startswith("sql"):
                    comp = comp[3:]
                comp = comp.strip()
            step = requests.post(
                f"{BASE_URL}/step",
                json={"action_type": "submit_query", "sql_query": comp},
                timeout=15,
            ).json()
            score = float(step.get("observation", {}).get("partial_score", 0.0))
            scores.append(score)
            print(f"  Eval {i+1}/{n_episodes} [{task_id}]: score={score:.3f}")
        except Exception as e:
            print(f"  Eval {i+1}/{n_episodes} failed: {e}")
            scores.append(0.0)
    mean = sum(scores) / len(scores) if scores else 0.0
    return mean, scores


try:
    baseline_mean, baseline_scores = evaluate_model(model, tokenizer, NUM_EVAL_EPISODES)
    print(f"\\nUntrained mean score: {baseline_mean:.4f}")
except Exception as e:
    print("Baseline eval failed:", e)
    baseline_mean, baseline_scores = 0.0, []
"""
)


# ─── Cell 8 — Train ──────────────────────────────────────────────────────────
md(
    """## Step 6 — Configure and Run GRPO Training

GRPO samples 4 completions per prompt, uses the live env's `partial_score` as reward,
and updates only the LoRA adapters (the 4-bit base remains frozen)."""
)
code(
    """
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOTrainer, GRPOConfig

dataset = Dataset.from_dict({
    "prompt": [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": p["prompt"]},
        ]
        for p in prompts_meta
    ],
    "task_id": [p["task_id"] for p in prompts_meta],
})

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
    bf16=False, fp16=True,
    gradient_checkpointing=True,
    use_vllm=False,
)

train_rewards, train_losses = [], []

class MetricCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        for key in ("reward", "rewards/reward_fn", "train/reward"):
            if key in logs:
                train_rewards.append(float(logs[key])); break
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

try:
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model + LoRA adapters saved to {OUTPUT_DIR}")
except Exception as e:
    print("Training failed:", e)
    raise
"""
)


# ─── Cell 9 — Post-training eval ─────────────────────────────────────────────
md(
    """## Step 7 — Post-Training Evaluation

Re-run the same 10-episode evaluation against the trained policy and print a
side-by-side comparison."""
)
code(
    """
try:
    trained_mean, trained_scores = evaluate_model(trainer.model, tokenizer, NUM_EVAL_EPISODES)
    delta = trained_mean - baseline_mean
    base_succ  = sum(s >= 0.5 for s in baseline_scores) / max(1, len(baseline_scores))
    train_succ = sum(s >= 0.5 for s in trained_scores)  / max(1, len(trained_scores))
    print("\\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Metric':<32} {'Baseline':>10} {'Trained':>10} {'Delta':>10}")
    print("-"*60)
    print(f"{'Mean episode score':<32} {baseline_mean:>10.4f} {trained_mean:>10.4f} {delta:>+10.4f}")
    print(f"{'Success rate (score>=0.5)':<32} {base_succ:>10.2%} {train_succ:>10.2%} {train_succ - base_succ:>+10.2%}")
    print("="*60)
except Exception as e:
    print("Post-training eval failed:", e)
    trained_mean, trained_scores = 0.0, []
"""
)


# ─── Cell 10 — Plots ─────────────────────────────────────────────────────────
md(
    """## Step 8 — Visualize Results

Save the three required plots to disk *and* render them inline:
- `training_reward_curve.png`
- `training_loss_curve.png`
- `before_after_comparison.png`"""
)
code(
    """
import matplotlib.pyplot as plt

def save_plots(train_rewards, train_losses, baseline_mean, trained_mean):
    if train_rewards:
        steps = list(range(len(train_rewards)))
        plt.figure(figsize=(10, 5))
        plt.plot(steps, train_rewards, "b-", linewidth=1.5, alpha=0.5, label="Per-step reward")
        window = max(1, len(train_rewards) // 20)
        smoothed = [sum(train_rewards[max(0, i-window):i+1])/max(1, min(i+1, window+1))
                    for i in range(len(train_rewards))]
        plt.plot(steps, smoothed, "r-", linewidth=2.5, label=f"Rolling mean (window={window})")
        plt.xlabel("Training Step"); plt.ylabel("Episode Reward (partial_score)")
        plt.title("Training Reward Curve — DataOps Incident Response Env\\nQwen2.5-1.5B + GRPO")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("training_reward_curve.png", dpi=150)
        plt.show()

    if train_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(list(range(len(train_losses))), train_losses, "g-", linewidth=1.5)
        plt.xlabel("Training Step"); plt.ylabel("Policy Loss")
        plt.title("Training Loss Curve — DataOps Incident Response Env\\nQwen2.5-1.5B + GRPO")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("training_loss_curve.png", dpi=150)
        plt.show()

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Untrained\\n(Qwen2.5-1.5B)", f"Trained\\n(GRPO, {NUM_TRAIN_PROMPTS} eps)"],
        [baseline_mean, trained_mean],
        color=["#e74c3c", "#2ecc71"], width=0.5, edgecolor="black", linewidth=1.2,
    )
    for bar, score in zip(bars, [baseline_mean, trained_mean]):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f"{score:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    plt.ylim(0, max(1.1, max(baseline_mean, trained_mean) * 1.2))
    plt.ylabel("Mean Episode Score (partial_score)")
    plt.title("Before vs After Training\\nDataOps Incident Response Env — Qwen2.5-1.5B + GRPO")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig("before_after_comparison.png", dpi=150)
    plt.show()

try:
    save_plots(train_rewards, train_losses, baseline_mean, trained_mean)
except Exception as e:
    print("Plotting failed:", e)
"""
)


# ─── Cell 11 — Push to HF Hub ────────────────────────────────────────────────
md(
    """## Step 9 — Save Model to HF Hub (Optional)

If `HF_TOKEN` is set in Cell 2, push the LoRA adapters + tokenizer to your
account on the Hugging Face Hub. The 4-bit base model is *not* uploaded
(it's loaded on-demand from `Qwen/Qwen2.5-1.5B-Instruct`)."""
)
code(
    """
import json

results = {
    "model": MODEL_NAME,
    "base_url": BASE_URL,
    "baseline_score": baseline_mean,
    "trained_score": trained_mean,
    "improvement": trained_mean - baseline_mean,
    "baseline_scores": baseline_scores,
    "trained_scores": trained_scores,
    "num_train_prompts": NUM_TRAIN_PROMPTS,
    "train_logged_steps": len(train_rewards),
}
with open("training_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved training_results.json")

if HF_TOKEN:
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
        print("HF Hub push failed:", e)
else:
    print("HF_TOKEN not set — skipping push.")
"""
)


nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python", "version": "3.11"},
}

OUT = "train_grpo.ipynb"
with open(OUT, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {OUT}  ({len(cells)} cells)")
