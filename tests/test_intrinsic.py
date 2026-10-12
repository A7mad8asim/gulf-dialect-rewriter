import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("intrinsic", Path(__file__).resolve().parents[1] / "eval" / "intrinsic.py")
intrinsic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intrinsic)


def test_score_uses_every_reference():
    pytest.importorskip("sacrebleu")  # in the eval extra, not needed for the rest of the tests
    refs = [["كيف حالك اليوم يا صديقي العزيز", "ما حالك اليوم يا صديقي العزيز", "كيف أنت اليوم يا صديقي"]]
    assert intrinsic.score(["ما حالك اليوم يا صديقي العزيز"], refs)["bleu"] == 100.0  # the second reference exactly
    assert intrinsic.score(["شي ثاني تماما"], refs)["bleu"] < 10
