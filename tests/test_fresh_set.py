"""The fresh dialect test set is frozen: these checks keep it well-formed (its SQL is checked against
ask-the-data's database by eval/extrinsic.py, which needs that repository)."""

import json
from pathlib import Path

from gdr.clinic import generate
from gdr.guard import check
from gdr.leakage import LeakageChecker

FRESH = Path(__file__).resolve().parents[1] / "eval" / "fresh_dialect.jsonl"


def _items():
    return [json.loads(line) for line in FRESH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_fields_and_ids():
    items = _items()
    assert len(items) == 25 and len({i["id"] for i in items}) == 25
    for i in items:
        assert {"id", "lang", "dialect", "difficulty", "question", "msa", "sql"} <= i.keys()
        assert i["lang"] == "ar" and i["dialect"] and i["difficulty"] in ("easy", "medium", "hard")
        assert i["sql"].lstrip().upper().startswith("SELECT")


def test_reference_msa_respects_the_design_rule():
    assert all(check(i["question"], i["msa"]).accepted for i in _items())


def test_no_question_is_close_to_generated_training_questions():
    checker = LeakageChecker([p["source"] for p in generate(3000, seed=13)])
    assert not [i["id"] for i in _items() if checker.leaks(i["question"])]
