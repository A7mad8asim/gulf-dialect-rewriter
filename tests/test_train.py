import json

import pytest

from gdr.rewrite import SYSTEM_PROMPT
from gdr.train import collate, corpus_counts, encode, load_pairs, messages


class FakeTokenizer:
    """Characters as tokens, with a chat template shaped like Qwen3's (thinking off)."""

    eos_token = "|"

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, enable_thinking):
        assert add_generation_prompt and not tokenize and enable_thinking is False
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_prompt_is_the_inference_prompt():
    assert messages("كم موعد تكنسل؟") == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "كم موعد تكنسل؟"},
    ]


def test_only_the_answer_and_end_token_are_trained_on():
    tok = FakeTokenizer()
    ex = encode(tok, {"source": "وش", "target": "ما"}, max_length=10_000)
    trained = "".join(chr(t) for t, label in zip(ex["input_ids"], ex["labels"]) if label != -100)
    assert trained == "ما|"
    assert "".join(chr(t) for t in ex["input_ids"]).endswith("<user>وش<assistant>ما|")


def test_too_long_pairs_are_dropped_not_cut():
    assert encode(FakeTokenizer(), {"source": "وش", "target": "ما"}, max_length=5) is None


def test_collate_pads_inputs_labels_and_mask():
    batch = collate([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}, {"input_ids": [4], "labels": [4]}], pad_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_open_only_drops_research_only_pairs(tmp_path):
    _write(tmp_path / "a.jsonl", [
        {"source": "x", "target": "y", "corpus": "dial2msa", "open_licence": False},
        {"source": "x", "target": "y", "corpus": "sts_saudi", "open_licence": True},
    ])
    _write(tmp_path / "b.jsonl", [{"source": "x", "target": "y", "corpus": "clinic", "open_licence": True}])
    assert corpus_counts(load_pairs(tmp_path, ("a.jsonl", "b.jsonl"), open_only=False)) == {"dial2msa": 1, "sts_saudi": 1, "clinic": 1}
    assert corpus_counts(load_pairs(tmp_path, ("a.jsonl", "b.jsonl"), open_only=True)) == {"sts_saudi": 1, "clinic": 1}


def test_missing_data_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="gdr fetch"):
        load_pairs(tmp_path, ("train.jsonl",), open_only=False)
