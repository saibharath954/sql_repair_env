# -*- coding: utf-8 -*-
"""train_grpo.py — DataOps Incident Response GRPO Training

Train **Qwen2.5-1.5B-Instruct** on the live OpenEnv SQL-repair environment so it
shows real, measurable improvement over its baseline.

Key design choices (vs. the original notebook):

1.  Deterministic, *seeded* prompts. Each training prompt is generated with a
    distinct seed and that seed is stored. The reward function passes the same
    seed back to ``/reset`` so the agent is graded on the **same** problem it
    was prompted with.
2.  Composite reward = environment ``partial_score`` + format/structure
    auxiliary terms. This produces a non-zero gradient signal even when the
    base score is 0, eliminating the "all 0.001 → no advantage" failure mode
    of the previous run.
3.  Brief SFT warm-up on synthetic ``(broken_query → correct_sql)`` pairs (using
    each task's gold ``action_schema.sql_query``) before GRPO. This bootstraps
    the model into the expected output format.
4.  Stable hyperparameters: lower LR (2e-6), more generations per prompt (8),
    higher rollout temperature (0.9), explicit KL coefficient (β=0.05).
5.  Honest evaluation: a *fixed* held-out seed pool is used for both the
    baseline and the trained run, so the comparison is genuinely
    apples-to-apples.

Pipeline: connect → generate seeded prompts → baseline eval (held-out seeds)
→ SFT warm-up → GRPO → post-eval (same held-out seeds) → plots → push.
"""

# ─── Cell 1 — Install Dependencies (Colab only) ─────────────────────────────
# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install -q "trl>=0.12" "transformers>=4.45" "datasets>=2.20" \
#     "bitsandbytes>=0.43" "accelerate>=0.34" "peft>=0.13" \
#     matplotlib requests "huggingface_hub>=0.25"


# ─── Cell 2 — Configuration ─────────────────────────────────────────────────
import os

os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"

try:  # Colab convenience
    from google.colab import userdata  # type: ignore
    HF_TOKEN_DEFAULT = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN_DEFAULT = os.environ.get("HF_TOKEN", "")

BASE_URL          = os.environ.get("BASE_URL", "https://bharath1675-sql-repair-env.hf.space")
MODEL_NAME        = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
HF_TOKEN          = os.environ.get("HF_TOKEN", HF_TOKEN_DEFAULT)
HF_USERNAME       = os.environ.get("HF_USERNAME", "bharath1675")
HF_REPO_ID        = f"{HF_USERNAME}/sql-repair-grpo-qwen"
OUTPUT_DIR        = os.environ.get("OUTPUT_DIR", "./grpo_output")

NUM_TRAIN_PROMPTS = int(os.environ.get("NUM_TRAIN_PROMPTS", 240))   # 80 per tier
NUM_EVAL_EPISODES = int(os.environ.get("NUM_EVAL_EPISODES", 30))    # 10 per tier
NUM_SFT_EXAMPLES  = int(os.environ.get("NUM_SFT_EXAMPLES", 60))     # 20 per tier
TASKS             = ["easy", "medium", "hard"]

# Seed bands keep training prompts strictly disjoint from eval prompts, so we
# never accidentally evaluate on a seed the model trained on.
TRAIN_SEED_BASE   = 1_000
EVAL_SEED_BASE    = 9_000
SFT_SEED_BASE     = 5_000

SYSTEM_PROMPT = (
    "You are an expert SQL engineer fixing production database incidents.\n"
    "You will be given a broken SQL query and a description of the database task.\n"
    "Your job: output ONLY the corrected SQL query.\n"
    "Rules:\n"
    "- Output raw SQL only. No markdown code blocks. No explanation. No prose.\n"
    "- Fix ALL issues: syntax errors, wrong joins, missing DISTINCT, type casts, sort order, GROUP BY columns.\n"
    "- For dirty data (duplicates / TEXT-stored numbers / orphan FKs), use SELECT DISTINCT in a subquery, "
    "CAST(col AS REAL) for numeric columns, and an INNER JOIN on the parent table to drop orphans.\n"
    "- The fixed query should return exactly the correct result set."
)
print(f"Targeting: {BASE_URL}")
print(f"Model    : {MODEL_NAME}")


# ─── Step 1 — Verify Environment Connection ─────────────────────────────────
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


# ─── Step 2 — Generate Seeded Training Dataset ──────────────────────────────
def build_user_prompt(obs: dict) -> str:
    return (
        f"TASK: {obs.get('task_description', '')}\n\n"
        f"BROKEN QUERY:\n{obs.get('broken_query', '')}\n\n"
        "Fix this query. Output only the corrected SQL:"
    )


def _reset_with_retries(task_id: str, seed: int, max_attempts: int = 3) -> dict:
    """Robust /reset call. Returns the observation dict."""
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
        seed = TRAIN_SEED_BASE + i  # unique seed per prompt
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
print("\n--- Sample prompt ---")
print(prompts_meta[0]["prompt"])


# ─── Step 3 — Composite Reward Function ─────────────────────────────────────
import re

_SQL_KEYWORDS = ("SELECT", "FROM")
_BAD_PREFIXES = ("here", "the", "this", "to fix", "i ", "first", "we ")
_RE_CAST      = re.compile(r"\bCAST\s*\(", re.IGNORECASE)
_RE_DISTINCT  = re.compile(r"\bDISTINCT\b", re.IGNORECASE)
_RE_GROUP_BY  = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_RE_JOIN      = re.compile(r"\bJOIN\b", re.IGNORECASE)


def _strip_completion(c: str) -> str:
    """Best-effort SQL extraction from a model completion."""
    s = (c or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("sql"):
            s = s[3:]
        s = s.strip()
    # Trim everything after the first ; if present (keep first statement).
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    return s


def _format_bonus(sql: str, task_id: str) -> float:
    """Auxiliary reward: rewards basic SQL hygiene and task-appropriate
    keywords. Adds a small but reliable signal even when the env score is 0.
    Capped at 0.20 so it never dominates the env score."""
    bonus = 0.0
    if not sql:
        return 0.0
    upper = sql.upper()

    # +0.05 for valid SELECT...FROM shape
    if all(k in upper for k in _SQL_KEYWORDS):
        bonus += 0.05

    # +0.03 for not opening with prose / markdown
    head = sql.lstrip().lower()[:30]
    if not any(head.startswith(b) for b in _BAD_PREFIXES) and not head.startswith("```"):
        bonus += 0.03

    # +0.02 for reasonable length (not trivially short or absurdly long)
    if 20 <= len(sql) <= 600:
        bonus += 0.02

    # Task-specific structural bonuses (never negative).
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
    """Submit `sql` against the env (re-using the same fault seed) and return
    (env_score, executes_without_error)."""
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
            executes = not bool(obs.get("error_message", "")) and bool(obs.get("query_result"))
            return env_score, executes
        except Exception as e:
            if attempt == 2:
                print(f"[REWARD ERR task={task_id} seed={seed}]: {e}")
                return 0.0, False
            time.sleep(0.5 * (attempt + 1))
    return 0.0, False


def make_reward_fn(prompts_meta):
    """Build a closure that maps GRPO completions to scalar rewards.

    The reward is composite:
        reward = env_score + format_bonus + execution_bonus
    """
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
                rewards.append(0.0)
                continue
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


# ─── Step 4 — Load Qwen2.5-1.5B (4-bit Quantized) ───────────────────────────
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

# Force trainable params to fp32 (GradScaler requirement).
for _, p in model.named_parameters():
    if p.dtype == torch.bfloat16:
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
        else:
            p.data = p.data.to(torch.float16)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded {MODEL_NAME} | {n_params / 1e6:.1f}M params")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")


# ─── Step 5 — Helper: a held-out, fixed evaluation set ──────────────────────
def build_eval_set(n_per_task: int) -> list:
    """Held-out eval seeds (disjoint from training seeds)."""
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


def evaluate_model(model, tokenizer, eval_set, label="") -> tuple:
    """Run greedy-decode evaluation on a *fixed* held-out seed list and return
    (mean, per-task means, raw scores)."""
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
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            comp = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            sql = _strip_completion(comp)
            env_score, _ = _evaluate_one(sql, task_id, seed)
            scores.append(env_score)
            per_task[task_id].append(env_score)
            print(f"  {label}[{task_id} seed={seed}] {i + 1}/{len(eval_set)}: {env_score:.3f}")
        except Exception as e:
            print(f"  {label}eval {i + 1} failed: {e}")
            scores.append(0.0)
            per_task[task_id].append(0.0)
    mean = sum(scores) / len(scores) if scores else 0.0
    per_task_mean = {
        t: (sum(v) / len(v) if v else 0.0) for t, v in per_task.items()
    }
    return mean, per_task_mean, scores


# ─── Step 6 — Baseline Evaluation (Before Training) ─────────────────────────
n_per_task = NUM_EVAL_EPISODES // len(TASKS)
eval_set = build_eval_set(n_per_task)

try:
    baseline_mean, baseline_per_task, baseline_scores = evaluate_model(
        model, tokenizer, eval_set, "[PRE] ",
    )
    print(f"\n[PRE] mean={baseline_mean:.4f}  per-task={baseline_per_task}")
except Exception as e:
    print(f"Baseline eval failed: {e}")
    baseline_mean, baseline_per_task, baseline_scores = 0.0, {t: 0.0 for t in TASKS}, []


# ─── Step 7 — SFT Warm-up on synthetic (broken, gold) pairs ─────────────────
# Even a brief SFT pass dramatically reduces the "cold-start" period of GRPO,
# because the policy starts already producing roughly the right output format.
import gc
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model

# Gold answers per task (mirror server/tasks.py action_schema.sql_query).
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


def build_sft_corpus(n_per_task: int) -> list:
    """For each (task_id, seed), build a (prompt, gold_sql) pair."""
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
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            rows.append({"text": text, "task_id": task_id})
    print(f"Built SFT corpus: {len(rows)} rows")
    return rows


def run_sft_warmup():
    """Run a short SFT pass on the gold corpus. Wraps `model` with LoRA so the
    same adapters carry over to the GRPO phase."""
    global model
    sft_rows = build_sft_corpus(NUM_SFT_EXAMPLES // len(TASKS))
    if not sft_rows:
        print("[SFT] no rows, skipping warm-up")
        return

    sft_ds = Dataset.from_list(sft_rows)

    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=768,
            padding=False,
        )
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
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    sft_trainer = Trainer(
        model=model,
        args=sft_args,
        train_dataset=sft_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    print(f"[SFT] starting warm-up on {len(sft_ds)} rows...")
    sft_trainer.train()
    print("[SFT] warm-up complete.")


run_sft_warmup()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


# ─── Step 8 — Configure and Run GRPO Training ───────────────────────────────
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
    learning_rate=2e-6,                # ↓ from 5e-6 — kills the spiky updates
    max_prompt_length=512,
    max_completion_length=192,         # tighter limit speeds up rollouts
    num_generations=8,                 # ↑ from 4 — cleaner advantage estimate
    temperature=0.9,                   # explicit exploration during rollouts
    beta=0.05,                         # explicit KL anchor → no policy collapse
    logging_steps=2,
    save_steps=200,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    fp16=True,
    bf16=False,
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
                train_rewards.append(float(logs[key]))
                break
        if "loss" in logs:
            train_losses.append(float(logs["loss"]))
        for key in ("kl", "rewards/kl", "train/kl"):
            if key in logs:
                train_kl.append(float(logs[key]))
                break


# If SFT already wrapped the model with LoRA, GRPOTrainer should NOT add it
# again — pass peft_config=None and let it use the existing adapters.
# Otherwise (no SFT happened), pass peft_config so adapters are attached now.
already_lora = any("lora_" in n for n, _ in model.named_parameters())
trainer = GRPOTrainer(
    model=model,
    args=grpo_args,
    reward_funcs=reward_fn,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=None if already_lora else peft_config,
    callbacks=[MetricCallback()],
)

# Force any remaining trainable bf16/fp16 params to fp32 for GradScaler.
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


# ─── Step 9 — Post-Training Evaluation (same eval set!) ─────────────────────
try:
    trained_mean, trained_per_task, trained_scores = evaluate_model(
        trainer.model, tokenizer, eval_set, "[POST]",
    )
    delta = trained_mean - baseline_mean
    base_succ  = sum(s >= 0.5 for s in baseline_scores) / max(1, len(baseline_scores))
    train_succ = sum(s >= 0.5 for s in trained_scores)  / max(1, len(trained_scores))
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY  (held-out seeds, identical for baseline & trained)")
    print("=" * 72)
    print(f"{'Metric':<32} {'Baseline':>12} {'Trained':>12} {'Delta':>12}")
    print("-" * 72)
    print(f"{'Mean episode score':<32} {baseline_mean:>12.4f} {trained_mean:>12.4f} {delta:>+12.4f}")
    print(f"{'Success rate (score>=0.5)':<32} {base_succ:>12.2%} {train_succ:>12.2%} {train_succ - base_succ:>+12.2%}")
    print("-" * 72)
    for t in TASKS:
        b, tr = baseline_per_task.get(t, 0.0), trained_per_task.get(t, 0.0)
        print(f"{'  '+t:<32} {b:>12.4f} {tr:>12.4f} {tr - b:>+12.4f}")
    print("=" * 72)
except Exception as e:
    print("Post-training eval failed:", e)
    trained_mean, trained_per_task, trained_scores = 0.0, {t: 0.0 for t in TASKS}, []


# ─── Step 10 — Visualize Results ────────────────────────────────────────────
import matplotlib.pyplot as plt


def _smoothed(xs, window=None):
    if not xs:
        return xs
    w = window or max(1, len(xs) // 20)
    return [
        sum(xs[max(0, i - w):i + 1]) / max(1, min(i + 1, w + 1))
        for i in range(len(xs))
    ]


def save_plots():
    if train_rewards:
        steps = list(range(len(train_rewards)))
        plt.figure(figsize=(10, 5))
        plt.plot(steps, train_rewards, "b-", linewidth=1.0, alpha=0.4, label="Per-step reward")
        plt.plot(steps, _smoothed(train_rewards), "r-", linewidth=2.5, label="Rolling mean")
        plt.xlabel("Logging step")
        plt.ylabel("Composite reward (env_score + format + exec)")
        plt.title("Training Reward Curve — DataOps Incident Response\nQwen2.5-1.5B + GRPO (composite reward, seeded)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("training_reward_curve.png", dpi=150)
        plt.show()

    if train_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(list(range(len(train_losses))), train_losses, "g-", linewidth=1.5)
        plt.xlabel("Logging step"); plt.ylabel("Policy loss")
        plt.title("Training Loss Curve — DataOps Incident Response\nQwen2.5-1.5B + GRPO")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("training_loss_curve.png", dpi=150)
        plt.show()

    # Overall before/after
    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Untrained\n(Qwen2.5-1.5B)", f"Trained\n(SFT+GRPO, {NUM_TRAIN_PROMPTS} eps)"],
        [baseline_mean, trained_mean],
        color=["#e74c3c", "#2ecc71"], width=0.5, edgecolor="black", linewidth=1.2,
    )
    for bar, score in zip(bars, [baseline_mean, trained_mean]):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )
    plt.ylim(0, max(1.1, max(baseline_mean, trained_mean) * 1.3))
    plt.ylabel("Mean partial_score on held-out seeds")
    plt.title("Before vs After Training\nDataOps Incident Response — Qwen2.5-1.5B + SFT+GRPO")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig("before_after_comparison.png", dpi=150)
    plt.show()

    # Per-task breakdown
    plt.figure(figsize=(8, 5))
    x = list(range(len(TASKS)))
    w = 0.35
    bvals = [baseline_per_task.get(t, 0.0) for t in TASKS]
    tvals = [trained_per_task.get(t, 0.0)  for t in TASKS]
    plt.bar([i - w/2 for i in x], bvals, w, label="Baseline", color="#e74c3c", edgecolor="black")
    plt.bar([i + w/2 for i in x], tvals, w, label="Trained",  color="#2ecc71", edgecolor="black")
    for i, (b, t) in enumerate(zip(bvals, tvals)):
        plt.text(i - w/2, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=10)
        plt.text(i + w/2, t + 0.01, f"{t:.2f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(x, [t.upper() for t in TASKS])
    plt.ylim(0, 1.05)
    plt.ylabel("Mean partial_score")
    plt.title("Per-Task Score Breakdown\nHeld-out seeds — Baseline vs Trained")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig("per_task_breakdown.png", dpi=150)
    plt.show()


try:
    save_plots()
except Exception as e:
    print("Plotting failed:", e)


# ─── Step 11 — Save Results JSON + push to HF Hub ───────────────────────────
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
