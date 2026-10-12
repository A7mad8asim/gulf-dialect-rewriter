"""QLoRA fine-tuning of a small Qwen3 model as a Gulf-dialect → MSA rewriter.

Each pair becomes a chat example with the same system prompt the rewriter uses at inference
(rewrite.SYSTEM_PROMPT), the dialect question as the user turn and the MSA question as the
assistant turn, with thinking off. Only the assistant turn counts towards the loss.

It uses the plain transformers Trainer with its own tokenization: TRL's SFT trainer needs Triton for
its fused loss, and Triton ships with PyTorch on Linux only.

    gdr train                                  # Qwen3-1.7B, all training pairs
    gdr train --open-only                      # only the openly licensed pairs (no Dial2MSA)
    gdr train --max-steps 20 --name smoke      # quick check of the whole pipeline

Needs the `train` extra and a CUDA GPU. The adapter, tokenizer and run.json (settings, data counts,
losses, sample rewrites) go to outputs/<name>/, which is never committed.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .rewrite import SYSTEM_PROMPT, clean_output

TRAIN_FILES = ("train.jsonl", "clinic_train.jsonl")
DEV_FILES = ("dev.jsonl", "clinic_dev.jsonl")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen3-1.7B"
    name: str = "qwen3-1.7b-all"
    data_dir: str = "data/pairs"
    output_dir: str = "outputs"
    open_only: bool = False  # train only on openly licensed pairs (drops Dial2MSA-Verified)
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: float = 2.0
    max_steps: int = -1  # > 0 overrides epochs (for smoke tests)
    batch_size: int = 16
    grad_accum: int = 1
    max_length: int = 256
    warmup_ratio: float = 0.03
    eval_every: int = 200
    dev_sample: int = 600  # dev pairs scored for eval loss (all of them is slow and adds little)
    seed: int = 13
    samples: int = 8  # dev questions rewritten at the end, as a sanity check
    target_modules: list[str] = field(default_factory=lambda: list(TARGET_MODULES))


def load_pairs(data_dir: Path, files: tuple[str, ...], open_only: bool) -> list[dict]:
    pairs = []
    for name in files:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run `gdr fetch` and `gdr clinic` first")
        with open(path, encoding="utf-8") as f:
            pairs += [json.loads(line) for line in f if line.strip()]
    return [p for p in pairs if p["open_licence"] or not open_only]


def messages(source: str) -> list[dict]:
    """The prompt the rewriter sends at inference."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": source}]


def prompt_text(tokenizer, source: str) -> str:
    return tokenizer.apply_chat_template(messages(source), add_generation_prompt=True, tokenize=False, enable_thinking=False)


def encode(tokenizer, pair: dict, max_length: int) -> dict | None:
    """Token ids of prompt + answer, with labels only on the answer (and its end-of-turn token).

    The prompt is tokenized exactly as at inference, so the model learns to continue from the same
    tokens it will see. Pairs longer than max_length are dropped rather than cut.
    """
    prompt = tokenizer(prompt_text(tokenizer, pair["source"]), add_special_tokens=False)["input_ids"]
    answer = tokenizer(pair["target"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt) + len(answer) > max_length:
        return None
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


def pad_batch(batch: list[dict], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a batch: pad tokens in the inputs, -100 in the labels, 0 in the attention mask."""
    width = max(len(b["input_ids"]) for b in batch)
    pad = lambda xs, v: xs + [v] * (width - len(xs))  # noqa: E731
    return {
        "input_ids": [pad(b["input_ids"], pad_id) for b in batch],
        "labels": [pad(b["labels"], -100) for b in batch],
        "attention_mask": [pad([1] * len(b["input_ids"]), 0) for b in batch],
    }


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    return {k: torch.tensor(v) for k, v in pad_batch(batch, pad_id).items()}


def corpus_counts(pairs: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p["corpus"]] = counts.get(p["corpus"], 0) + 1
    return counts


def train(cfg: TrainConfig) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    data_dir, out = Path(cfg.data_dir), Path(cfg.output_dir) / cfg.name
    train_pairs = load_pairs(data_dir, TRAIN_FILES, cfg.open_only)
    dev_pairs = load_pairs(data_dir, DEV_FILES, cfg.open_only)
    rng = random.Random(cfg.seed)
    rng.shuffle(train_pairs)
    rng.shuffle(dev_pairs)
    eval_pairs = dev_pairs[: cfg.dev_sample]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    train_set = [e for e in (encode(tokenizer, p, cfg.max_length) for p in train_pairs) if e]
    eval_set = [e for e in (encode(tokenizer, p, cfg.max_length) for p in eval_pairs) if e]
    print(f"train {len(train_set)} of {len(train_pairs)} {corpus_counts(train_pairs)}  |  dev {len(eval_set)} of {len(dev_pairs)}")
    _show_example(tokenizer, train_set[0])

    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if cfg.load_in_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_ratio,  # a float below 1 is a ratio of the total steps (transformers 5)
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=cfg.eval_every,
        save_strategy="steps",
        save_steps=cfg.eval_every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=0,  # Windows
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    started = time.time()
    result = trainer.train()
    minutes = (time.time() - started) / 60
    final_eval = trainer.evaluate()
    trainer.model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")

    samples = _sample_rewrites(trainer.model, tokenizer, dev_pairs[-cfg.samples:])
    run = {
        "config": asdict(cfg),
        "data": {"train": corpus_counts(train_pairs), "train_encoded": len(train_set), "dev_scored": len(eval_set)},
        "train_loss": round(result.training_loss, 4),
        "eval_loss": round(final_eval["eval_loss"], 4),
        "steps": result.global_step,
        "minutes": round(minutes, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log_history": trainer.state.log_history,
        "samples": samples,
        "versions": _versions(),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ntrain loss {run['train_loss']}  eval loss {run['eval_loss']}  {run['steps']} steps  "
          f"{run['minutes']} min  peak GPU {run['peak_gpu_gb']} GB")
    for s in samples:
        print(f"  {s['source']}\n    -> {s['rewrite']}\n    ref {s['target']}")
    print(f"saved: {out / 'adapter'}")
    return run


def _show_example(tokenizer, example: dict) -> None:
    """Print one training sequence and the part that is trained on, to check the masking."""
    print("--- training sequence ---\n" + tokenizer.decode(example["input_ids"]))
    print("--- trained on ---\n" + tokenizer.decode([t for t, l in zip(example["input_ids"], example["labels"]) if l != -100]))
    print("-------------------------")


def _sample_rewrites(model, tokenizer, pairs: list[dict]) -> list[dict]:
    import torch

    model.eval()
    model.config.use_cache = True
    out = []
    for p in pairs:
        inputs = tokenizer(prompt_text(tokenizer, p["source"]), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        reply = tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"source": p["source"], "rewrite": clean_output(reply), "target": p["target"]})
    return out


def _versions() -> dict:
    import importlib.metadata as md

    return {pkg: md.version(pkg) for pkg in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")}
