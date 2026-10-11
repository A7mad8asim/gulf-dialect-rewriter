import json

from gdr.cli import main
from gdr.leakage import LeakageChecker

HELD_OUT = ["كم موعد كان عندنا في 2024؟", "How many patients are registered in the network?"]


def test_exact_match_after_normalization_leaks():
    c = LeakageChecker(HELD_OUT)
    assert c.leaks("كَم مَوعِد كان عندنا في ٢٠٢٤")


def test_near_duplicate_leaks_and_unrelated_does_not():
    c = LeakageChecker(HELD_OUT)
    assert c.leaks("كم موعد كان عندنا في 2023؟")
    assert not c.leaks("وين أقرب صيدلية؟")


def test_filter_checks_source_and_target():
    c = LeakageChecker(HELD_OUT)
    pairs = [
        {"source": "شلونك اليوم؟", "target": "كيف حالك اليوم؟"},
        {"source": "شكثر موعد عندنا 2024", "target": "كم موعد كان عندنا في 2024؟"},
    ]
    kept, dropped = c.filter_pairs(pairs)
    assert [p["source"] for p in kept] == ["شلونك اليوم؟"] and len(dropped) == 1


def test_cli_writes_kept_pairs(tmp_path, capsys):
    held = tmp_path / "gold.jsonl"
    held.write_text("\n".join(json.dumps({"question": q}, ensure_ascii=False) for q in HELD_OUT), encoding="utf-8")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps({"source": "شلونك؟", "target": "كيف حالك؟"}, ensure_ascii=False) + "\n"
        + json.dumps({"source": "كم موعد كان عندنا في 2024؟", "target": "x"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "kept.jsonl"
    assert main(["check-leakage", str(pairs), "--held-out", str(held), "--out", str(out)]) == 0
    assert "1 kept, 1 overlap" in capsys.readouterr().out
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
