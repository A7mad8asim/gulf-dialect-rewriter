"""The fresh dialect test set is frozen: these checks keep it well-formed (its SQL is checked against
ask-the-data's database by eval/extrinsic.py, which needs that repository)."""

import json
from pathlib import Path

from gdr.cli import main
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


def test_clinic_build_drops_pairs_close_to_the_fresh_set(tmp_path):
    # The generator was expanded after the fresh set was written, so `gdr clinic` must hold it out
    held = tmp_path / "gold.jsonl"
    held.write_text(json.dumps({"question": "placeholder"}), encoding="utf-8")
    assert main(["clinic", "--n", "6000", "--data-dir", str(tmp_path), "--held-out", str(held)]) == 0
    checker = LeakageChecker([i["question"] for i in _items()])
    kept = [json.loads(line)["source"] for name in ("clinic_train.jsonl", "clinic_dev.jsonl")
            for line in (tmp_path / "pairs" / name).read_text(encoding="utf-8").splitlines()]
    assert kept and not [s for s in kept if checker.leaks(s)]
