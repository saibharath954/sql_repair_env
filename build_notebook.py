"""
Programmatic builder for train_grpo.ipynb.

Run:
    python build_notebook.py        # writes train_grpo.ipynb in the same dir

This builder is the *single source of truth* for what the notebook does. It
mirrors `train_grpo.py` cell-for-cell so we never drift between the two.
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
    """# DataOps Incident Response — GRPO Training Notebook (v2)

Train **Qwen2.5-1.5B-Instruct** to diagnose and fix corrupted production
databases. The reward signal is the live OpenEnv environment's `partial_score`,
combined with a small format/structure bonus to keep the gradient non-zero
even early in training.

**What's new in v2 vs v1**
- Deterministic *seeded* prompts: the broken_query the agent is graded on is
  the same one it was prompted with.
- Composite reward = `env_score + format_bonus + execute_bonus`. No more
  collapsed advantages from rows of `0.001`.
- Brief SFT warm-up on synthetic gold answers before GRPO.
- Stable hyperparameters (`lr=2e-6`, `num_generations=8`, `temperature=0.9`,
  explicit KL anchor `β=0.05`).
- Held-out evaluation set with **identical seeds** for baseline and trained
  runs, so before/after numbers are genuinely apples-to-apples.

Pipeline: connect → seeded prompts → baseline eval → SFT warm-up → GRPO →
post-eval → plots → push.
"""
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

os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"

try:
    from google.colab import userdata
    HF_TOKEN_DEFAULT = userdata.get('HF_TOKEN')
except Exception:
    HF_TOKEN_DEFAULT = os.environ.get("HF_TOKEN", "")

BASE_URL          = "https://bharath1675-sql-repair-env.hf.space"
MODEL_NAME        = "Qwen/Qwen2.5-1.5B-Instruct"
HF_TOKEN          = HF_TOKEN_DEFAULT
HF_USERNAME       = "bharath1675"
HF_REPO_ID        = f"{HF_USERNAME}/sql-repair-grpo-qwen"
OUTPUT_DIR        = "./grpo_output"

NUM_TRAIN_PROMPTS = 240    # 80 per tier
NUM_EVAL_EPISODES = 30     # 10 per tier
NUM_SFT_EXAMPLES  = 60     # 20 per tier
TASKS             = ["easy", "medium", "hard"]

# Disjoint seed bands for train / sft / eval — eval is held out.
TRAIN_SEED_BASE = 1_000
SFT_SEED_BASE   = 5_000
EVAL_SEED_BASE  = 9_000

SYSTEM_PROMPT = (
    "You are an expert SQL engineer fixing production database incidents.\\n"
    "You will be given a broken SQL query and a description of the database task.\\n"
    "Your job: output ONLY the corrected SQL query.\\n"
    "Rules:\\n"
    "- Output raw SQL only. No markdown code blocks. No explanation. No prose.\\n"
    "- Fix ALL issues: syntax errors, wrong joins, missing DISTINCT, type casts, sort order, GROUP BY columns.\\n"
    "- For dirty data (duplicates / TEXT-stored numbers / orphan FKs), use SELECT DISTINCT in a subquery, "
    "CAST(col AS REAL) for numeric columns, and an INNER JOIN on the parent table to drop orphans.\\n"
    "- The fixed query should return exactly the correct result set."
)
print(f"Targeting: {BASE_URL}")
print(f"Model    : {MODEL_NAME}")
"""
)


# ─── Cell 3 — Verify environment ─────────────────────────────────────────────
md(
    """## Step 1 — Verify Environment Connection

Sanity-check the live HF Space is reachable and `/reset` honours the seed
parameter (we rely on this for reproducible training)."""
)
code(
    """
import requests
import time

health = requests.get(f"{BASE_URL}/health", timeout=15)
health.raise_for_status()
print("HEALTH:", health.json())

reset = requests.post(f"{BASE_URL}/reset", json={"task_id": "easy", "seed": 0}, timeout=15)
reset.raise_for_status()
obs0 = reset.json()["observation"]
print("Task :", obs0.get("task_description", "")[:200])
print("Query:", obs0.get("broken_query", "")[:200])
"""
)


# ─── Cell 4 — Seeded dataset ─────────────────────────────────────────────────
md(
    """## Step 2 — Generate Seeded Training Dataset

Each prompt is associated with a unique seed. The reward function later
re-uses the **same seed** so the agent is graded on the exact problem it was
prompted with."""
)
code(
    """
def build_user_prompt(obs: dict) -> str:
    return (
        f"TASK: {obs.get('task_description', '')}\\n\\n"
        f"BROKEN QUERY:\\n{obs.get('broken_query', '')}\\n\\n"
        "Fix this query. Output only the corrected SQL:"
    )


def _reset_with_retries(task_id: str, seed: int, max_attempts: int = 3) -> dict:
    for attempt in range(max_attempts):
        try:
            r = requests.post(
                f"{BASE_URL}/reset",
                json={"task_id": task_id, "seed": int(seed)},
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("observation", {})
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"[warn] /reset({task_id}, seed={seed}) failed: {e}")
                return {}
            time.sleep(0.5 * (attempt + 1))
    return {}


def generate_training_prompts():
    prompts = []
    cycle = TASKS * (NUM_TRAIN_PROMPTS // len(TASKS) + 1)
    for i, task_id in enumerate(cycle[:NUM_TRAIN_PROMPTS]):
        seed = TRAIN_SEED_BASE + i
        obs = _reset_with_retries(task_id, seed)
        if not obs:
            continue
        prompts.append({
            "prompt":       build_user_prompt(obs),
            "task_id":      task_id,
            "broken_query": obs.get("broken_query", ""),
            "seed":         seed,
        })
    print(f"Generated {len(prompts)} seeded training prompts")
    return prompts


prompts_meta = generate_training_prompts()
print("\\n--- Sample prompt ---")
print(prompts_meta[0]["prompt"])
"""
)


# ─── Cell 5 — Composite reward function ──────────────────────────────────────
md(
    """## Step 3 — Composite Reward Function

```
reward = env_partial_score + format_bonus + execute_bonus
```

`format_bonus` and `execute_bonus` are small (≤ 0.20 and 0.05) so the env
score still dominates, but they keep the gradient non-zero so GRPO has
meaningful advantages even when the env score is 0.
"""
)
code(
    """
import re

_SQL_KEYWORDS = ("SELECT", "FROM")
_BAD_PREFIXES = ("here", "the", "this", "to fix", "i ", "first", "we ")
_RE_CAST      = re.compile(r"\\bCAST\\s*\\(", re.IGNORECASE)
_RE_DISTINCT  = re.compile(r"\\bDISTINCT\\b", re.IGNORECASE)
_RE_GROUP_BY  = re.compile(r"\\bGROUP\\s+BY\\b", re.IGNORECASE)
_RE_JOIN      = re.compile(r"\\bJOIN\\b", re.IGNORECASE)


def _strip_completion(c: str) -> str:
    s = (c or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("sql"):
            s = s[3:]
        s = s.strip()
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    return s


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


def _evaluate_one(sql: str, task_id: str, seed: int) -> tuple:
    if not sql:
        return 0.0, False
    for attempt in range(3):
        try:
            r = requests.post(
                f"{BASE_URL}/reset",
                json={"task_id": task_id, "seed": int(seed)},
                timeout=30,
            )
            r.raise_for_status()
            s = requests.post(
                f"{BASE_URL}/step",
                json={"action": {"action_type": "submit_query", "sql_query": sql}},
                timeout=30,
            )
            s.raise_for_status()
            obs = s.json().get("observation", {})
            env_score = float(obs.get("partial_score", 0.0))
            executes  = not bool(obs.get("error_message", "")) and bool(obs.get("query_result"))
            return env_score, executes
        except Exception as e:
            if attempt == 2:
                print(f"[REWARD ERR task={task_id} seed={seed}]: {e}")
                return 0.0, False
            time.sleep(0.5 * (attempt + 1))
    return 0.0, False


def make_reward_fn(prompts_meta):
    str_to_meta = {p["prompt"]: p for p in prompts_meta}
    counter = {"n": 0}

    def _extract_prompt_str(p) -> str:
        if isinstance(p, list):
            for msg in p:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
            return ""
        return str(p)

    def reward_fn(completions, prompts=None, **kwargs):
        flat = [
            "".join(t.get("content", "") for t in c) if isinstance(c, list) else str(c)
            for c in completions
        ]
        flat_prompts = [_extract_prompt_str(p) for p in (prompts or [""] * len(flat))]
        rewards = []
        for comp, ps in zip(flat, flat_prompts):
            meta = str_to_meta.get(ps)
            if meta is None:
                rewards.append(0.0); continue
            tid, seed = meta["task_id"], meta["seed"]
            sql = _strip_completion(comp)
            env_score, executes = _evaluate_one(sql, tid, seed)
            fmt = _format_bonus(sql, tid)
            exec_bonus = 0.05 if executes else 0.0
            r = env_score + fmt + exec_bonus
            counter["n"] += 1
            print(
                f"Ep {counter['n']:>4}: task={tid} seed={seed} "
                f"env={env_score:.3f} fmt={fmt:.2f} exec={exec_bonus:.2f} → r={r:.3f}",
                flush=True,
            )
            rewards.append(r)
        return rewards

    return reward_fn


reward_fn = make_reward_fn(prompts_meta)
test_reward = reward_fn(
    ["SELECT name, salary FROM employees WHERE dept='Engineering' ORDER BY salary DESC"],
    prompts=[prompts_meta[0]["prompt"]],
)
print(f"Sanity reward: {test_reward}")
"""
)


# ─── Cell 6 — Load model ─────────────────────────────────────────────────────
md(
    """## Step 4 — Load Qwen2.5-1.5B (4-bit Quantized)

NF4 4-bit + QLoRA on attention + MLP projections. Fits a T4 (~16 GB)."""
)
code(
    """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

for _, p in model.named_parameters():
    if p.dtype == torch.bfloat16:
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
        else:
            p.data = p.data.to(torch.float16)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded {MODEL_NAME} | {n_params/1e6:.1f}M params")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
"""
)


# ─── Cell 7 — Held-out eval set + baseline ───────────────────────────────────
md(
    """## Step 5 — Held-out Evaluation Set + Baseline

We build a *fixed* eval set with seeds disjoint from the training set, and
re-use it both for the baseline and the post-training run. This is the
honest-comparison invariant."""
)
code(
    """
def build_eval_set(n_per_task):
    eval_set = []
    for tier_idx, task_id in enumerate(TASKS):
        for k in range(n_per_task):
            seed = EVAL_SEED_BASE + 1000 * tier_idx + k
            obs = _reset_with_retries(task_id, seed)
            if not obs:
                continue
            eval_set.append({
                "task_id":      task_id,
                "seed":         seed,
                "broken_query": obs.get("broken_query", ""),
                "task_desc":    obs.get("task_description", ""),
                "prompt":       build_user_prompt(obs),
            })
    print(f"Built held-out eval set: {len(eval_set)} episodes "
          f"({n_per_task} per task)")
    return eval_set


def evaluate_model(model, tokenizer, eval_set, label=""):
    model.eval()
    scores, per_task = [], {t: [] for t in TASKS}
    for i, ex in enumerate(eval_set):
        task_id, seed = ex["task_id"], ex["seed"]
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": ex["prompt"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    temperature=None, top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            comp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            sql = _strip_completion(comp)
            env_score, _ = _evaluate_one(sql, task_id, seed)
            scores.append(env_score)
            per_task[task_id].append(env_score)
            print(f"  {label}[{task_id} seed={seed}] {i+1}/{len(eval_set)}: {env_score:.3f}")
        except Exception as e:
            print(f"  {label}eval {i+1} failed: {e}")
            scores.append(0.0); per_task[task_id].append(0.0)
    mean = sum(scores) / len(scores) if scores else 0.0
    per_task_mean = {t: (sum(v)/len(v) if v else 0.0) for t, v in per_task.items()}
    return mean, per_task_mean, scores


n_per_task = NUM_EVAL_EPISODES // len(TASKS)
eval_set = build_eval_set(n_per_task)

try:
    baseline_mean, baseline_per_task, baseline_scores = evaluate_model(
        model, tokenizer, eval_set, "[PRE] ",
    )
    print(f"\\n[PRE] mean={baseline_mean:.4f}  per-task={baseline_per_task}")
except Exception as e:
    print(f"Baseline eval failed: {e}")
    baseline_mean, baseline_per_task, baseline_scores = 0.0, {t: 0.0 for t in TASKS}, []
"""
)


# ─── Cell 8 — SFT warm-up ────────────────────────────────────────────────────
md(
    """## Step 6 — SFT Warm-up on synthetic (broken, gold) pairs

Even ~60 supervised pairs (using each task's known-correct SQL as the
target) get the model into roughly the right output format and shape. After
this warm-up GRPO is no longer cold-starting from "the model has no clue"."""
)
code(
    """
import gc
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model

GOLD_SQL = {
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


def build_sft_corpus(n_per_task):
    rows = []
    for tier_idx, task_id in enumerate(TASKS):
        for k in range(n_per_task):
            seed = SFT_SEED_BASE + 1000 * tier_idx + k
            obs = _reset_with_retries(task_id, seed)
            if not obs:
                continue
            prompt = build_user_prompt(obs)
            messages = [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": GOLD_SQL[task_id]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            rows.append({"text": text, "task_id": task_id})
    print(f"Built SFT corpus: {len(rows)} rows")
    return rows


def run_sft_warmup():
    global model
    sft_rows = build_sft_corpus(NUM_SFT_EXAMPLES // len(TASKS))
    if not sft_rows:
        print("[SFT] no rows, skipping warm-up"); return

    sft_ds = Dataset.from_list(sft_rows)

    def _tok(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=768, padding=False)
        out["labels"] = [ids.copy() for ids in out["input_ids"]]
        return out

    sft_ds = sft_ds.map(_tok, batched=True, remove_columns=sft_ds.column_names)

    model = get_peft_model(model, peft_config)
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
            p.data = p.data.to(torch.float32)

    sft_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "sft"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=1e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=True, bf16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    sft_trainer = Trainer(
        model=model, args=sft_args,
        train_dataset=sft_ds,
        tokenizer=tokenizer, data_collator=collator,
    )
    print(f"[SFT] starting warm-up on {len(sft_ds)} rows...")
    sft_trainer.train()
    print("[SFT] warm-up complete.")


run_sft_warmup()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
"""
)


# ─── Cell 9 — GRPO training ──────────────────────────────────────────────────
md(
    """## Step 7 — GRPO Training

Stable hyperparameters: `lr=2e-6`, `num_generations=8`, `temperature=0.9`,
explicit KL anchor `β=0.05`. Samples 8 completions per prompt and updates
LoRA adapters using the composite reward."""
)
code(
    """
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
    "_task_id": [p["task_id"] for p in prompts_meta],
    "_seed":    [p["seed"]    for p in prompts_meta],
})

grpo_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-6,
    max_prompt_length=512,
    max_completion_length=192,
    num_generations=8,
    temperature=0.9,
    beta=0.05,
    logging_steps=2,
    save_steps=200,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    fp16=True, bf16=False,
    gradient_checkpointing=True,
    use_vllm=False,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
)

train_rewards, train_losses, train_kl = [], [], []


class MetricCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        for key in (
            "reward", "rewards/reward_fn", "train/reward",
            "mean_reward", "rewards/mean", "rewards/reward_fn/mean",
        ):
            if key in logs:
                train_rewards.append(float(logs[key])); break
        if "loss" in logs:
            train_losses.append(float(logs["loss"]))
        for key in ("kl", "rewards/kl", "train/kl"):
            if key in logs:
                train_kl.append(float(logs[key])); break


already_lora = any("lora_" in n for n, _ in model.named_parameters())
trainer = GRPOTrainer(
    model=model, args=grpo_args,
    reward_funcs=reward_fn,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=None if already_lora else peft_config,
    callbacks=[MetricCallback()],
)

for _, p in trainer.model.named_parameters():
    if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
        p.data = p.data.to(torch.float32)

try:
    trainer.train()
    try:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(OUTPUT_DIR)
        print("LoRA merged and saved.")
    except Exception as me:
        trainer.model.save_pretrained(OUTPUT_DIR)
        print(f"Saved adapters only (merge failed: {me})")
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")
except Exception as e:
    print(f"Training error: {e}")
    raise
"""
)


# ─── Cell 10 — Post-training eval ────────────────────────────────────────────
md(
    """## Step 8 — Post-Training Evaluation (same eval set!)

We re-use the **identical** held-out eval set from Step 5, so the comparison
is genuinely apples-to-apples."""
)
code(
    """
try:
    trained_mean, trained_per_task, trained_scores = evaluate_model(
        trainer.model, tokenizer, eval_set, "[POST]",
    )
    delta = trained_mean - baseline_mean
    base_succ  = sum(s >= 0.5 for s in baseline_scores) / max(1, len(baseline_scores))
    train_succ = sum(s >= 0.5 for s in trained_scores)  / max(1, len(trained_scores))
    print("\\n" + "="*72)
    print("RESULTS SUMMARY  (held-out seeds, identical for baseline & trained)")
    print("="*72)
    print(f"{'Metric':<32} {'Baseline':>12} {'Trained':>12} {'Delta':>12}")
    print("-"*72)
    print(f"{'Mean episode score':<32} {baseline_mean:>12.4f} {trained_mean:>12.4f} {delta:>+12.4f}")
    print(f"{'Success rate (score>=0.5)':<32} {base_succ:>12.2%} {train_succ:>12.2%} {train_succ - base_succ:>+12.2%}")
    print("-"*72)
    for t in TASKS:
        b, tr = baseline_per_task.get(t, 0.0), trained_per_task.get(t, 0.0)
        print(f"{'  '+t:<32} {b:>12.4f} {tr:>12.4f} {tr - b:>+12.4f}")
    print("="*72)
except Exception as e:
    print("Post-training eval failed:", e)
    trained_mean, trained_per_task, trained_scores = 0.0, {t: 0.0 for t in TASKS}, []
"""
)


# ─── Cell 11 — Plots ─────────────────────────────────────────────────────────
md(
    """## Step 9 — Visualize Results

Saves four plots:
- `training_reward_curve.png`
- `training_loss_curve.png`
- `before_after_comparison.png`
- `per_task_breakdown.png`"""
)
code(
    """
import matplotlib.pyplot as plt


def _smoothed(xs, window=None):
    if not xs: return xs
    w = window or max(1, len(xs) // 20)
    return [sum(xs[max(0, i-w):i+1])/max(1, min(i+1, w+1)) for i in range(len(xs))]


def save_plots():
    if train_rewards:
        steps = list(range(len(train_rewards)))
        plt.figure(figsize=(10, 5))
        plt.plot(steps, train_rewards, "b-", linewidth=1.0, alpha=0.4, label="Per-step reward")
        plt.plot(steps, _smoothed(train_rewards), "r-", linewidth=2.5, label="Rolling mean")
        plt.xlabel("Logging step"); plt.ylabel("Composite reward (env_score + format + exec)")
        plt.title("Training Reward Curve — DataOps Incident Response\\nQwen2.5-1.5B + GRPO (composite reward, seeded)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("training_reward_curve.png", dpi=150); plt.show()

    if train_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(list(range(len(train_losses))), train_losses, "g-", linewidth=1.5)
        plt.xlabel("Logging step"); plt.ylabel("Policy loss")
        plt.title("Training Loss Curve — DataOps Incident Response\\nQwen2.5-1.5B + GRPO")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("training_loss_curve.png", dpi=150); plt.show()

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Untrained\\n(Qwen2.5-1.5B)", f"Trained\\n(SFT+GRPO, {NUM_TRAIN_PROMPTS} eps)"],
        [baseline_mean, trained_mean],
        color=["#e74c3c", "#2ecc71"], width=0.5, edgecolor="black", linewidth=1.2,
    )
    for bar, score in zip(bars, [baseline_mean, trained_mean]):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f"{score:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    plt.ylim(0, max(1.1, max(baseline_mean, trained_mean) * 1.3))
    plt.ylabel("Mean partial_score on held-out seeds")
    plt.title("Before vs After Training\\nDataOps Incident Response — Qwen2.5-1.5B + SFT+GRPO")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig("before_after_comparison.png", dpi=150); plt.show()

    plt.figure(figsize=(8, 5))
    x = list(range(len(TASKS))); w = 0.35
    bvals = [baseline_per_task.get(t, 0.0) for t in TASKS]
    tvals = [trained_per_task.get(t, 0.0)  for t in TASKS]
    plt.bar([i - w/2 for i in x], bvals, w, label="Baseline", color="#e74c3c", edgecolor="black")
    plt.bar([i + w/2 for i in x], tvals, w, label="Trained",  color="#2ecc71", edgecolor="black")
    for i, (b, t) in enumerate(zip(bvals, tvals)):
        plt.text(i - w/2, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=10)
        plt.text(i + w/2, t + 0.01, f"{t:.2f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(x, [t.upper() for t in TASKS]); plt.ylim(0, 1.05)
    plt.ylabel("Mean partial_score")
    plt.title("Per-Task Score Breakdown\\nHeld-out seeds — Baseline vs Trained")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig("per_task_breakdown.png", dpi=150); plt.show()


try:
    save_plots()
except Exception as e:
    print("Plotting failed:", e)
"""
)


# ─── Cell 12 — Save + push ───────────────────────────────────────────────────
md(
    """## Step 10 — Save Results JSON + push to HF Hub (Optional)"""
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
    "baseline_per_task": baseline_per_task,
    "trained_per_task": trained_per_task,
    "num_train_prompts": NUM_TRAIN_PROMPTS,
    "num_eval_episodes": len(eval_set),
    "num_sft_examples":  NUM_SFT_EXAMPLES,
    "train_logged_steps": len(train_rewards),
    "eval_seed_base": EVAL_SEED_BASE,
    "train_seed_base": TRAIN_SEED_BASE,
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
            commit_message="GRPO+SFT QLoRA adapters — composite reward, seeded eval",
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
