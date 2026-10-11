from gdr.data import Pair, dedupe, filter_pairs, parse_dial2msa_csv, parse_dial2msa_test, parse_sts_rows
from gdr.leakage import LeakageChecker

DIAL2MSA_CSV = (
    "id,msa_verified,tweet,cleanedtweet2\n"
    "14.0,تعلموا كيف تتحدثون يا أحبائي,تعلموا اشلون تسولفون يا احباب 😁,تعلموا اشلون تسولفون يا احباب\n"
    "14.0,تعلموا كيف تتكلمون يا أحباب,تعلموا اشلون تسولفون يا احباب 😁,تعلموا اشلون تسولفون يا احباب\n"
    "15.0,,وش تبون,وش تبون\n"
)


def test_dial2msa_keeps_every_reference_and_skips_empty_rows():
    pairs = parse_dial2msa_csv(DIAL2MSA_CSV, "train")
    assert [p.target for p in pairs] == ["تعلموا كيف تتحدثون يا أحبائي", "تعلموا كيف تتكلمون يا أحباب"]
    assert all(p.source == "تعلموا اشلون تسولفون يا احباب" and p.corpus == "dial2msa" for p in pairs)
    assert pairs[0].to_json()["open_licence"] is False


def test_dial2msa_test_groups_references_by_line():
    items = parse_dial2msa_test(["وش تبون", "اشلونك"], [["ماذا تريدون", "كيف حالك"], ["ماذا تريدون؟", ""]])
    assert items[0]["references"] == ["ماذا تريدون", "ماذا تريدون؟"]
    assert items[1]["references"] == ["كيف حالك"]


def test_sts_rows_give_two_pairs_and_skip_blank_or_multiline_cells():
    rows = [
        ("فتاة تصفف شعرها", "بنت تزين شعرها.", "فتاة تمشط شعرها", "بنت تكد شعرها."),
        ("ثلاث فتيات يجلسن على العشب.", "ثلاث بنات يجلسون على زرع.", "كلب يركض.", None),
        ("سطر\nسطر آخر", "سطر\nسطر", None, None),
    ]
    pairs = parse_sts_rows(rows, first=0)
    assert [(p.source, p.target) for p in pairs] == [
        ("بنت تزين شعرها.", "فتاة تصفف شعرها"),
        ("بنت تكد شعرها.", "فتاة تمشط شعرها"),
        ("ثلاث بنات يجلسون على زرع.", "ثلاث فتيات يجلسن على العشب."),
    ]
    assert pairs[0].to_json()["open_licence"] is True


def test_sts_test_sheet_starts_after_the_score_column():
    pairs = parse_sts_rows([("0.2", "رجل يركل رجلا آخر.", "رجال يشوت رجال ثاني.", "", "")], first=1)
    assert [(p.source, p.target) for p in pairs] == [("رجال يشوت رجال ثاني.", "رجل يركل رجلا آخر.")]


def test_dedupe_ignores_diacritics_and_punctuation():
    a = Pair("وش تبون", "ماذا تريدون", "dial2msa", "1")
    b = Pair("وش تبون؟", "ماذا تُريدون", "dial2msa", "2")
    assert dedupe([a, b]) == [a]


def test_filter_drops_guard_failures_and_held_out_overlap():
    pairs = [
        Pair("الفاشلون قسمان 1 فكر ولم يفعل 2 فعل ولم يفكر", "الفاشلون قسمان فكر ولم يفعل وقسم فعل ولم يفكر", "dial2msa", "1"),
        Pair("كم موعد كان عندنا في 2024؟", "كم عدد المواعيد في عام 2024؟", "dial2msa", "2"),
        Pair("وش تبون", "ماذا تريدون", "dial2msa", "3"),
    ]
    kept, rejected = filter_pairs(pairs, LeakageChecker(["كم موعد كان عندنا في 2024؟"]))
    assert [p.id for p in kept] == ["3"]
    assert [r["reason"].split(":")[0] for r in rejected] == ["guard", "overlaps held-out text"]
