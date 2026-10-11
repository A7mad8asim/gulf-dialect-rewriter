import json

from gdr.clinic import GROUPS, MEASURES, generate
from gdr.cli import main
from gdr.guard import check


def test_same_seed_same_pairs_and_different_seed_differs():
    assert generate(200, seed=1) == generate(200, seed=1)
    assert generate(200, seed=1) != generate(200, seed=2)


def test_sources_are_distinct_and_every_kind_appears():
    pairs = generate(1500, seed=3)
    assert len({p["source"] for p in pairs}) == len(pairs) == 1500
    assert {p["kind"] for p in pairs} == {"plain", "grouped", "top", "identity"}


def test_every_pair_respects_the_design_rule():
    # The generator itself must never teach the model to drop or invent a number
    assert all(check(p["source"], p["target"]).accepted for p in generate(3000, seed=4))


def test_targets_are_msa_questions_and_no_template_slot_is_left():
    for p in generate(2000, seed=5):
        assert p["target"].endswith("؟")
        assert "{" not in p["source"] + p["target"]
        if p["kind"] == "identity":
            assert p["source"] == p["target"]
        else:
            assert p["source"] != p["target"]


def test_superlatives_only_use_groupings_that_can_be_ranked():
    ranked = {g for g, spec in GROUPS.items() if spec[2]}
    assert ranked and all(set(m.groups) & ranked for m in MEASURES if m.top)


def test_cli_writes_train_and_dev_without_held_out_overlap(tmp_path, capsys):
    held = tmp_path / "gold.jsonl"
    held.write_text(json.dumps({"question": "كم موعد تكنسل في 2024؟"}, ensure_ascii=False), encoding="utf-8")
    assert main(["clinic", "--n", "400", "--seed", "7", "--data-dir", str(tmp_path), "--held-out", str(held)]) == 0
    stats = json.loads((tmp_path / "pairs" / "clinic_stats.json").read_text(encoding="utf-8"))
    train = (tmp_path / "pairs" / "clinic_train.jsonl").read_text(encoding="utf-8").splitlines()
    dev = (tmp_path / "pairs" / "clinic_dev.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train) == stats["train"] and len(dev) == stats["dev"] > 0
    assert stats["train"] + stats["dev"] + sum(stats["rejected"].values()) == 400
    sources = {json.loads(line)["source"] for line in train + dev}
    assert "كم موعد تكنسل في 2024؟" not in sources
    assert all(json.loads(line)["open_licence"] for line in train)
