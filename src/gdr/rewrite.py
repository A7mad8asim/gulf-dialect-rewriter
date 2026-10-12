"""Rewrite a question into MSA with a model served by Ollama, then apply the guard.

The prompted (not fine-tuned) base model is the baseline the fine-tuned adapter has to beat; the
adapter will be served the same way, so both go through this one code path and the same guard.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .guard import GuardResult, check

SYSTEM_PROMPT = (
    "You rewrite questions about a clinic network's data from Gulf-dialect Arabic into Modern Standard Arabic "
    "(الفصحى).\n"
    "Rules:\n"
    "- Keep the meaning exactly: same measure, filters, time period and grouping.\n"
    "- Keep every number, year, clinic name and Latin-script term unchanged.\n"
    "- Do not answer the question and do not add information.\n"
    "- If the question is already in Modern Standard Arabic, return it unchanged.\n"
    "Output only the rewritten question, nothing else."
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class Rewrite:
    question: str
    raw: str  # what the model returned, cleaned
    guard: GuardResult
    latency_s: float

    @property
    def text(self) -> str:  # what goes downstream
        return self.guard.text


def clean_output(text: str) -> str:
    """First non-empty line of the reply, without thinking blocks, labels or wrapping quotes."""
    text = _THINK.sub("", text or "").strip()
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = re.sub(r"^(?:MSA|الفصحى|السؤال(?: بالفصحى)?)\s*[:：]\s*", "", line)
    return line.strip().strip("\"'«»“”").strip()


class OllamaRewriter:
    def __init__(self, model: str = "qwen3:8b", base_url: str = "http://127.0.0.1:11434", timeout: float = 120):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def _chat(self, question: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
            "stream": False,
            "options": {"temperature": 0, "seed": 7},
        }
        if self.model.lower().startswith("qwen3"):
            payload["think"] = False
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]

    def rewrite(self, question: str) -> Rewrite:
        started = time.perf_counter()
        raw = clean_output(self._chat(question))
        return Rewrite(question, raw, check(question, raw), time.perf_counter() - started)


class AdapterRewriter:
    """A base model plus a LoRA adapter from `gdr train`, run with transformers (needs the train extra)."""

    def __init__(self, adapter_dir: str, max_new_tokens: int = 128):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from .train import TrainConfig

        run = json.loads((Path(adapter_dir).parent / "run.json").read_text(encoding="utf-8"))
        base = run["config"]["model"]
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        ) if run["config"].get("load_in_4bit", TrainConfig.load_in_4bit) else None
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0})
        self.model = PeftModel.from_pretrained(model, adapter_dir).eval()
        self.max_new_tokens, self._name = max_new_tokens, f"adapter:{Path(adapter_dir).parent.name}"

    @property
    def name(self) -> str:
        return self._name

    def _chat(self, question: str) -> str:
        import torch

        from .train import prompt_text

        inputs = self.tokenizer(prompt_text(self.tokenizer, question), return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    rewrite = OllamaRewriter.rewrite


SUBSETS = {  # which questions of a file to rewrite (ask-the-data's gold set mixes English, MSA and dialect)
    "all": lambda i: True,
    "arabic": lambda i: i.get("lang", "ar") == "ar",
    "dialect": lambda i: i.get("lang", "ar") == "ar" and bool(i.get("dialect")),
    "msa": lambda i: i.get("lang", "ar") == "ar" and not i.get("dialect"),
}


def rewrite_file(rewriter, in_path: str, out_path: str, subset: str = "all") -> list[dict]:
    """Rewrite the questions of a JSONL file (id, question); writes id, question, rewrite, latency."""
    with open(in_path, encoding="utf-8") as f:
        items = [i for i in (json.loads(line) for line in f if line.strip()) if SUBSETS[subset](i)]
    rows = []
    for i in items:
        r = rewriter.rewrite(i["question"])
        rows.append({"id": i["id"], "question": i["question"], "rewrite": r.raw, "latency_s": round(r.latency_s, 3)})
        print(f"  {i['id']:<7} {i['question']}\n          -> {r.raw}" + ("" if r.guard.accepted else f"   [guard: {'; '.join(r.guard.reasons)}]"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"rewriter": rewriter.name, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows
