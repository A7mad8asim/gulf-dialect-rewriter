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
