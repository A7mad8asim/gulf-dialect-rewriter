from gdr.rewrite import OllamaRewriter, clean_output


def test_clean_output_keeps_only_the_question():
    assert clean_output("<think>\n</think>\n\nكم عدد المواعيد في عام 2024؟\n") == "كم عدد المواعيد في عام 2024؟"
    assert clean_output("الفصحى: «كم عدد المواعيد؟»") == "كم عدد المواعيد؟"
    assert clean_output("") == ""


def test_rewrite_goes_through_the_guard(monkeypatch):
    rewriter = OllamaRewriter()
    monkeypatch.setattr(rewriter, "_chat", lambda q: "كم عدد مرضى السكري؟")
    r = rewriter.rewrite("كم مريض سكر عندنا فوق 60 سنة؟")
    assert not r.guard.accepted and r.text == "كم مريض سكر عندنا فوق 60 سنة؟"

    monkeypatch.setattr(rewriter, "_chat", lambda q: "كم عدد مرضى السكري الذين تزيد أعمارهم على 60 سنة؟")
    assert rewriter.rewrite("كم مريض سكر عندنا فوق 60 سنة؟").guard.accepted
