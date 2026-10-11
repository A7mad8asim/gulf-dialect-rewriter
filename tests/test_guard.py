from gdr.guard import check


def test_accepts_wording_change():
    r = check("كم موعد كان عندنا في 2024؟", "كم عدد المواعيد في عام 2024؟")
    assert r.accepted and r.text == "كم عدد المواعيد في عام 2024؟"


def test_arabic_digits_count_as_the_same_number():
    assert check("كم موعد كان عندنا في ٢٠٢٤؟", "كم عدد المواعيد في عام 2024؟").accepted


def test_lost_number_falls_back_to_question():
    q = "كم مريض سكري عندنا فوق 60 سنة في 2025؟"
    r = check(q, "كم عدد مرضى السكري في عام 2025؟")
    assert not r.accepted and r.text == q
    assert any("numbers lost: 60" in reason for reason in r.reasons)


def test_added_number_is_rejected():
    r = check("كم موعد فات الشهر اللي طاف؟", "كم عدد المواعيد الفائتة في شهر 9؟")
    assert not r.accepted and any("numbers added: 9" in reason for reason in r.reasons)


def test_latin_terms_must_survive():
    r = check("شكثر المرضى اللي HbA1c عندهم فوق 7؟", "ما عدد المرضى الذين تزيد نسبة السكر التراكمي لديهم عن 7؟")
    assert not r.accepted and any("terms lost: hba1c" in reason for reason in r.reasons)


def test_repeated_number_must_keep_its_count():
    assert not check("قارن 2024 مع 2024", "قارن عام 2024").accepted


def test_empty_and_runaway_rewrites_are_rejected():
    assert not check("وش عدد المراجعين؟", "  ").accepted
    assert not check("وش عدد المراجعين؟", "ما عدد المراجعين؟ " * 5).accepted
