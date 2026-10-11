"""Seeded generator of clinic-analytics questions in Gulf dialect, paired with Modern Standard Arabic.

The public corpora are tweets; ask-the-data's users ask كم questions about appointments, visits and
lab results. This generator fills that gap. Its vocabulary comes from ask-the-data's schema and
glossary (tables, departments, clinic names, Gulf words such as تكنسل, انصرف له, طلع), never from
its gold questions; any generated question too close to one of them is dropped by the leakage check.

A question is composed from a measure ("how many appointments", "no-show rate" ...), up to two
filters (sex, age, nationality, clinic, department, booking channel), an optional time filter
(year, month, Ramadan, summer ...) and either a grouping ("per clinic") or a superlative ("which
clinic had the most ..."). Every part has one MSA rendering and several Gulf renderings, so the
same seed always yields the same pairs. About one pair in ten is MSA → MSA, so the rewriter also
learns to leave questions that are already in MSA unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# --- filters: (MSA, [Gulf variants]) -------------------------------------------------------------

YEARS = (2023, 2024, 2025)
MONTHS = ("يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر")


def _time_filters(rng: random.Random, kind: str) -> tuple[str, str]:
    """One time filter, as (MSA, Gulf). `kind` is 'full' or 'year' (years only)."""
    y = rng.choice(YEARS)
    options = [
        (f"في عام {y}", [f"في {y}", f"سنة {y}", f"في سنة {y}", f"ب{y}"]),
        ("في هذا العام", ["هالسنة", "السنة هذي", "في هالسنة", "السنة هاذي"]),
        ("في العام الماضي", ["السنة اللي فاتت", "السنة اللي طافت", "السنة الماضية", "السنة اللي راحت", "العام اللي طاف"]),
    ]
    if kind == "full":
        m = rng.randrange(11)
        m2 = rng.randrange(m + 1, 12)
        q = rng.randint(1, 4)
        options += [
            (f"في شهر {MONTHS[m]} {y}", [f"في شهر {MONTHS[m]} {y}", f"في {MONTHS[m]} {y}", f"بشهر {MONTHS[m]} {y}"]),
            (f"في الشهر {m + 1} من عام {y}", [f"في شهر {m + 1} {y}", f"شهر {m + 1} سنة {y}"]),
            (f"خلال شهر رمضان {y}", [f"في رمضان {y}", f"برمضان {y}", f"في رمضان سنة {y}"]),
            (f"خلال صيف {y}", [f"في صيف {y}", f"بالصيف سنة {y}", f"في الصيف {y}"]),
            (f"خلال أيام العيد في عام {y}", [f"في العيد سنة {y}", f"أيام العيد {y}"]),
            ("في عطلة نهاية الأسبوع", ["في الويكند", "يوم الجمعة والسبت", "بالويكند"]),
            ("في الشهر الماضي", ["الشهر اللي طاف", "الشهر اللي فات", "الشهر الماضي"]),
            ("في الربع الأخير من العام", ["في آخر ربع من السنة", "بالربع الأخير"]),
            (f"في الربع {q} من عام {y}", [f"في الربع {q} من سنة {y}", f"بالربع {q} {y}"]),
            (f"خلال فصل الشتاء في عام {y}", [f"في الشتا سنة {y}", f"بالشتا {y}", f"في الشتاء {y}"]),
            (f"بين شهر {MONTHS[m]} وشهر {MONTHS[m2]} من عام {y}", [f"بين {MONTHS[m]} و{MONTHS[m2]} {y}", f"من {MONTHS[m]} لين {MONTHS[m2]} {y}"]),
            (f"من بداية عام {y} حتى نهاية {MONTHS[m2]}", [f"من أول {y} لين آخر {MONTHS[m2]}", f"من بداية {y} لين نهاية {MONTHS[m2]}"]),
        ]
    msa, gulf = rng.choice(options)
    return msa, rng.choice(gulf)


def _with_preposition(stem: str, people: bool) -> str:
    """'من الرجال' for counts of people, 'للرجال' for counts of events."""
    if people:
        return f"من {stem}"
    return f"لل{stem[2:]}" if stem.startswith("ال") else f"ل{stem}"


GROUPS_OF_PEOPLE = {  # dimension -> [(MSA stem, [Gulf stems])]
    "sex": [("الرجال", ["الرياييل", "الرجال", "الرياييل"]), ("النساء", ["الحريم", "النسوان", "الحرمات"])],
    "age": [("الأطفال", ["اليهال", "العيال", "البزارين", "الصغار", "الأطفال"]),
            ("كبار السن", ["الشيبان", "كبار السن", "الكبار", "الشياب"])],
    "nat": [("القطريين", ["القطريين", "المواطنين"]), ("العرب من غير القطريين", ["باقي العرب", "العرب غير القطريين"])],
}

DEPARTMENTS = [
    ("قسم الطب العام", ["الطب العام", "عيادة العام"]),
    ("عيادة السكري", ["عيادة السكر", "عيادة السكري"]),
    ("قسم صحة المرأة", ["عيادة الحريم", "صحة المرأة"]),
    ("قسم طب الأطفال", ["عيادة الأطفال", "عيادة اليهال"]),
    ("قسم الأسنان", ["عيادة الأسنان", "الأسنان"]),
    ("قسم التطعيمات", ["التطعيمات", "عيادة التطعيم"]),
]

WEEKDAYS = ("الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت")
HOURS = [  # (MSA, Gulf variants) for appointment times
    *[(f"قبل الساعة {h} صباحًا", [f"قبل الساعة {h} الصبح", f"قبل {h} الصبح"]) for h in (8, 9, 10)],
    *[(f"بعد الساعة {h} عصرًا", [f"بعد الساعة {h} العصر", f"بعد {h} العصر"]) for h in (3, 4, 5)],
    *[(f"بعد الساعة {h} مساءً", [f"بعد الساعة {h} بالليل", f"بعد {h} المسا"]) for h in (6, 7, 8)],
]

CLINICS = ("عيادة الدوحة المركزية", "عيادة الخليج الغربي", "عيادة الريان", "عيادة الوكرة",
           "عيادة أم صلال", "عيادة الخور", "عيادة المنطقة الصناعية", "عيادة الضعاين")

CHANNELS = [
    ("عن طريق التطبيق", ["من التطبيق", "عن طريق التطبيق", "بالتطبيق"]),
    ("عن طريق الهاتف", ["بالتلفون", "عن طريق التلفون", "بالتليفون"]),
    ("بالحضور المباشر دون حجز مسبق", ["للي جوا بدون حجز", "اللي جوا بدون موعد"]),
    ("عن طريق التحويل", ["بالتحويل", "عن طريق التحويل"]),
]


def _entity_filter(rng: random.Random, dim: str, people: bool) -> tuple[str, str]:
    if dim in GROUPS_OF_PEOPLE:
        msa, gulf = rng.choice(GROUPS_OF_PEOPLE[dim])
        return _with_preposition(msa, people), _with_preposition(rng.choice(gulf), people)
    if dim == "dept":
        msa, gulf = rng.choice(DEPARTMENTS)
        return f"في {msa}", rng.choice([f"في {g}" for g in gulf] + [f"ب{g}" for g in gulf])
    if dim == "clinic":
        name = rng.choice(CLINICS)
        short = name.removeprefix("عيادة ")
        return f"في {name}", rng.choice([f"في {name}", f"ب{name}", f"في {short}"])
    if dim == "channel":
        msa, gulf = rng.choice(CHANNELS)
        return msa, rng.choice(gulf)
    if dim == "weekday":
        day = rng.choice(WEEKDAYS)
        return f"يوم {day}", rng.choice([f"يوم {day}", f"أيام {day}"])
    if dim == "hour":
        msa, gulf = rng.choice(HOURS)
        return msa, rng.choice(gulf)
    raise ValueError(dim)


# --- groupings: dimension -> (definite, singular, 'fih' for superlatives, feminine) ----------------

GROUPS = {
    "clinic": ("العيادة", "عيادة", "فيها", True),
    "dept": ("القسم", "قسم", "فيه", False),
    "nat": ("الجنسية", "جنسية", "عندها", True),
    "age": ("الفئة العمرية", "فئة عمرية", "عندها", True),
    "month": ("الشهر", "شهر", "فيه", False),
    "year": ("السنة", "سنة", "فيها", True),
    "channel": ("طريقة الحجز", "طريقة حجز", None, True),
    "sex": ("الجنس", None, None, False),
    "weekday": ("يوم الأسبوع", "يوم", "فيه", False),
}


def _grouping(rng: random.Random, dim: str) -> tuple[str, str]:
    definite, singular, _, feminine = GROUPS[dim]
    msa = f"حسب {definite}"
    if dim == "sex":
        return msa, rng.choice(["رجال وحريم", "حسب الجنس", "للرياييل والحريم كل واحد بروحه"])
    alone = "بروحها" if feminine else "بروحه"
    return msa, rng.choice([f"لكل {singular}", f"في كل {singular}", f"حسب {definite}", f"على حسب {definite}", f"كل {singular} {alone}"])


# --- measures ------------------------------------------------------------------------------------

ALL_ENTITY = ("sex", "age", "nat", "clinic", "dept")


@dataclass(frozen=True)
class Measure:
    msa: str  # with {F} for filters and {G} for a grouping, both with a leading space when present
    gulf: tuple[str, ...]
    dims: tuple[str, ...]  # allowed entity filters
    people: bool  # counts people (من الرجال) rather than events (للرجال)
    time: str | None = "full"  # 'full', 'year' or None
    groups: tuple[str, ...] = ("clinic", "dept", "nat", "age", "month", "year", "sex")
    top: tuple[str, tuple[str, ...]] | None = None  # superlative phrase: MSA, Gulf variants
    low: tuple[str, tuple[str, ...]] | None = None  # the opposite superlative ("the fewest ..."), if it makes sense
    slots: tuple[tuple[str, tuple[str, ...]], ...] = field(default=())  # {X} fillers: MSA, Gulf variants


MEASURES = [
    Measure("كم عدد المواعيد{F}{G}؟", ("كم موعد عندنا{F}{G}", "كم موعد انحجز{F}{G}", "كم حجز صار{F}{G}", "شكثر المواعيد{F}{G}"),
            ALL_ENTITY + ("channel", "weekday", "hour"), False,
            groups=("clinic", "dept", "nat", "age", "month", "year", "sex", "channel", "weekday"),
            top=("أكبر عدد من المواعيد", ("أكثر شي مواعيد", "أكثر مواعيد")),
            low=("أقل عدد من المواعيد", ("أقل شي مواعيد", "أقل مواعيد"))),
    Measure("كم عدد المواعيد التي لم يحضرها المرضى{F}{G}؟", ("كم موعد ما حضروا له{F}{G}", "كم موعد ما جوا له المرضى{F}{G}"),
            ALL_ENTITY + ("channel", "weekday", "hour"), False, top=("أكبر عدد من حالات عدم الحضور", ("أكثر شي غياب", "أكثر ناس ما جوا"))),
    Measure("ما نسبة عدم الحضور إلى المواعيد{F}{G}؟", ("كم نسبة الغياب{F}{G}", "وش نسبة الغياب عن المواعيد{F}{G}", "نسبة اللي ما جوا مواعيدهم{F}{G} كم كانت"),
            ALL_ENTITY + ("channel", "weekday", "hour"), False,
            groups=("clinic", "dept", "nat", "age", "month", "year", "sex", "weekday"),
            top=("أعلى نسبة عدم حضور", ("أعلى نسبة غياب",)), low=("أدنى نسبة عدم حضور", ("أقل نسبة غياب",))),
    Measure("كم عدد المواعيد الملغاة{F}{G}؟", ("كم موعد تكنسل{F}{G}", "كم موعد انلغى{F}{G}", "كم حجز كنسلوه{F}{G}"),
            ("clinic", "dept", "channel"), False, top=("أكبر عدد من المواعيد الملغاة", ("أكثر شي مواعيد متكنسلة", "أكثر كنسلة"))),
    Measure("ما نسبة إلغاء المواعيد{F}{G}؟", ("كم نسبة الكنسلة{F}{G}", "وش نسبة المواعيد اللي تكنسلت{F}{G}"),
            ("clinic", "dept", "channel"), False),
    Measure("كم عدد الزيارات{F}{G}؟", ("كم زيارة صارت{F}{G}", "كم مراجعة صارت{F}{G}", "كم مرة راجعوا المرضى{F}{G}"),
            ALL_ENTITY + ("weekday",), False, top=("أكبر عدد من الزيارات", ("أكثر شي زيارات", "أكثر مراجعات")),
            low=("أقل عدد من الزيارات", ("أقل شي زيارات", "أقل مراجعات"))),
    Measure("كم عدد المرضى المختلفين الذين حضروا{F}{G}؟", ("كم مريض مختلف راجع{F}{G}", "كم واحد راجع{F}{G}", "كم مريض جا{F}{G}"),
            ALL_ENTITY, True),
    Measure("ما متوسط وقت الانتظار بالدقائق{F}{G}؟", ("كم معدل الانتظار بالدقايق{F}{G}", "كم دقيقة ينطر المريض بالمعدل{F}{G}", "كم يقعدون ينتظرون بالمعدل{F}{G}"),
            ("clinic", "dept", "weekday"), False, groups=("clinic", "dept", "month", "year", "weekday"),
            top=("أطول متوسط لوقت الانتظار", ("أطول انتظار", "أكثر انتظار"))),
    Measure("ما متوسط عدد الأيام بين الحجز والموعد{F}{G}؟", ("كم يوم ينطرون بالمعدل لين يجي الموعد{F}{G}", "كم معدل الأيام من الحجز للموعد{F}{G}"),
            ("clinic", "dept", "channel"), False, groups=("clinic", "dept", "channel", "year")),
    Measure("كم عدد المرضى الذين ظهر لديهم نقص في فيتامين د{F}{G}؟", ("كم واحد طلع عنده نقص فيتامين د{F}{G}", "كم مريض طلع عنده نقص فيتامين د{F}{G}"),
            ("sex", "age", "nat"), True, top=("أكبر عدد من المصابين بنقص فيتامين د", ("أكثر ناس عندهم نقص فيتامين د",))),
    Measure("ما متوسط نتيجة فيتامين د{F}{G}؟", ("كم معدل فيتامين د{F}{G}", "وش معدل نتايج فيتامين د{F}{G}"),
            ("sex", "age", "nat"), True, groups=("nat", "age", "sex", "year")),
    Measure("كم عدد مرضى السكري المسجلين{F}{G}؟", ("كم مريض سكر عندنا{F}{G}", "كم واحد عنده سكر{F}{G}", "كم مريض سكري مسجل{F}{G}"),
            ("sex", "age", "nat", "clinic"), True, time=None, groups=("clinic", "nat", "age", "sex"),
            top=("أكبر عدد من مرضى السكري", ("أكثر شي مرضى سكر", "أكثر مرضى سكر"))),
    Measure("كم عدد مرضى ارتفاع ضغط الدم المسجلين{F}{G}؟", ("كم مريض ضغط عندنا{F}{G}", "كم واحد عنده ضغط{F}{G}"),
            ("sex", "age", "nat", "clinic"), True, time=None, groups=("clinic", "nat", "age", "sex")),
    Measure("كم عدد المرضى الذين أجروا تحليل السكر التراكمي{F}{G}؟", ("كم واحد سوّى تحليل السكر التراكمي{F}{G}", "كم مريض سوا فحص التراكمي{F}{G}"),
            ("sex", "age", "nat"), True, groups=("nat", "age", "sex", "month", "year")),
    Measure("كم عدد المرضى الذين {V} لهم {X}{F}{G}؟", ("كم مريض {GV} له {X}{F}{G}", "كم واحد {GV} له {X}{F}{G}"),
            ("sex", "age", "nat"), True, groups=("nat", "age", "sex", "year"),
            slots=(("مكملات فيتامين د|وُصفت|انصرفت", ("مكملات فيتامين د",)), ("الإنسولين|وُصف|انصرف", ("انسولين", "الانسولين")),
                   ("المضادات الحيوية|وُصفت|انصرف", ("مضاد حيوي",)), ("الميتفورمين|وُصف|انصرف", ("ميتفورمين", "دوا الميتفورمين")),
                   ("أدوية الستاتين|وُصفت|انصرف", ("دوا الكوليسترول", "ستاتين")))),
    Measure("كم عدد نتائج {X} غير الطبيعية{F}{G}؟", ("كم {X} طلع غير طبيعي{F}{G}", "كم واحد طلع {X} حقه غير طبيعي{F}{G}"),
            ("sex", "age", "nat", "clinic"), False, groups=("clinic", "nat", "age", "year"),
            slots=(("فحص السكري", ("فحص سكر", "فحص السكر")), ("فحص ضغط الدم", ("فحص ضغط", "فحص الضغط")),
                   ("فحص سرطان الثدي", ("فحص سرطان الثدي",)), ("فحص سرطان القولون", ("فحص القولون", "فحص سرطان القولون")))),
    Measure("ما نسبة إكمال المتابعات المستحقة{F}{G}؟", ("كم نسبة المتابعات اللي خلصت{F}{G}", "كم نسبة اللي كملوا متابعاتهم{F}{G}"),
            ("clinic", "nat"), False, groups=("clinic", "nat", "year"), top=("أقل نسبة إكمال للمتابعات", ("أقل نسبة متابعات خلصت",))),
    Measure("كم عدد المرضى الجدد الذين سجلوا{F}{G}؟", ("كم مريض جديد تسجل{F}{G}", "كم واحد سجل جديد{F}{G}"),
            ("sex", "nat", "clinic"), True, time="year", groups=("clinic", "nat", "sex", "year"),
            top=("أكبر عدد من المرضى الجدد", ("أكثر شي مرضى جدد",))),
    # --- added after the first fine-tuned model learned the templates; drawn from the schema and glossary
    Measure("ما نسبة الحضور إلى المواعيد{F}{G}؟", ("كم نسبة الحضور{F}{G}", "وش نسبة اللي حضروا مواعيدهم{F}{G}", "نسبة الحضور{F}{G} كم كانت"),
            ALL_ENTITY + ("channel", "weekday"), False, top=("أعلى نسبة حضور", ("أعلى نسبة حضور", "أحسن حضور"))),
    Measure("كم عدد المرضى المسجلين{F}{G}؟", ("كم مريض مسجل عندنا{F}{G}", "كم واحد مسجل{F}{G}", "چم مريض مسجل{F}{G}"),
            ("sex", "age", "nat", "clinic"), True, time=None, groups=("clinic", "nat", "age", "sex"),
            top=("أكبر عدد من المرضى المسجلين", ("أكثر شي مرضى مسجلين",))),
    Measure("كم عدد مرضى {X} المسجلين{F}{G}؟", ("كم مريض {X} عندنا{F}{G}", "كم واحد عنده {X}{F}{G}", "كم مريض عنده {X}{F}{G}"),
            ("sex", "age", "nat", "clinic"), True, time=None, groups=("clinic", "nat", "age", "sex"),
            slots=(("الربو", ("ربو",)), ("ارتفاع الكوليسترول", ("كوليسترول",)), ("نقص فيتامين د", ("نقص فيتامين د",)))),
    Measure("كم عدد المرضى الذين شُخصوا ب{X} لأول مرة{F}{G}؟", ("كم واحد تشخص ب{X} أول مرة{F}{G}", "كم مريض جديد انشخص ب{X}{F}{G}"),
            ("sex", "age", "nat"), True, time="year", groups=("nat", "age", "sex", "year"),
            slots=(("السكري", ("السكر", "السكري")), ("ارتفاع ضغط الدم", ("الضغط",)), ("الربو", ("الربو",)),
                   ("ارتفاع الكوليسترول", ("الكوليسترول",)))),
    Measure("كم عدد المرضى الذين كان الكوليسترول الضار لديهم مرتفعًا{F}{G}؟",
            ("كم واحد طلع الكوليسترول الضار عنده مرتفع{F}{G}", "كم مريض عنده الكوليسترول الضار عالي{F}{G}"),
            ("sex", "age", "nat"), True, groups=("nat", "age", "sex", "year")),
    Measure("ما متوسط نتيجة الكوليسترول الضار{F}{G}؟", ("كم معدل الكوليسترول الضار{F}{G}", "وش معدل نتايج الكوليسترول الضار{F}{G}"),
            ("sex", "age", "nat"), True, groups=("nat", "age", "sex", "year")),
    Measure("ما متوسط نتيجة السكر التراكمي{F}{G}؟", ("كم معدل السكر التراكمي{F}{G}", "كم معدل التراكمي{F}{G}"),
            ("sex", "age", "nat"), True, groups=("nat", "age", "sex", "year")),
    Measure("كم عدد {X} التي أُجريت{F}{G}؟", ("كم {GX} انسوى{F}{G}", "كم {GX} سووا{F}{G}", "كم {GX} صار{F}{G}"),
            ("sex", "age", "nat", "clinic", "weekday"), False, groups=("clinic", "nat", "age", "year", "month"),
            slots=(("فحوصات السكري|فحص سكر", ()), ("فحوصات ضغط الدم|فحص ضغط", ()), ("فحوصات سرطان الثدي|فحص سرطان الثدي", ()),
                   ("فحوصات سرطان القولون|فحص قولون", ()))),
    Measure("ما نسبة النتائج غير الطبيعية في {X}{F}{G}؟", ("كم نسبة {GX} اللي طلع غير طبيعي{F}{G}", "وش نسبة {GX} غير الطبيعي{F}{G}"),
            ("sex", "age", "nat", "clinic"), False, groups=("clinic", "nat", "age", "year"),
            slots=(("فحوصات السكري|فحص السكر", ()), ("فحوصات ضغط الدم|فحص الضغط", ()), ("فحوصات سرطان الثدي|فحص سرطان الثدي", ()),
                   ("فحوصات سرطان القولون|فحص القولون", ()))),
    Measure("كم عدد المتابعات التي لم تكتمل وقد فات موعدها{F}{G}؟", ("كم متابعة ما خلصت وموعدها فات{F}{G}", "كم متابعة متأخرة ما تسوت{F}{G}"),
            ("clinic", "nat"), False, groups=("clinic", "nat", "year")),
    Measure("كم عدد المتابعات التي أُكملت في موعدها{F}{G}؟", ("كم متابعة خلصت في وقتها{F}{G}", "كم متابعة تسوت على وقتها{F}{G}"),
            ("clinic", "nat"), False, groups=("clinic", "nat", "year", "month")),
    Measure("كم عدد وصفات {X}{F}{G}؟", ("كم وصفة {GX} انصرفت{F}{G}", "كم مرة انصرف {GX}{F}{G}"),
            ("sex", "age", "nat", "weekday"), False, groups=("nat", "age", "sex", "year", "month"),
            slots=(("مكملات فيتامين د|مكملات فيتامين د", ()), ("الإنسولين|انسولين", ()), ("المضادات الحيوية|مضاد حيوي", ()),
                   ("الميتفورمين|ميتفورمين", ()), ("أدوية الستاتين|دوا الكوليسترول", ()), ("مسكنات الألم وخافضات الحرارة|بنادول", ()),
                   ("مضادات الهيستامين|دوا الحساسية", ()), ("بخاخات الكورتيزون|بخاخ الكورتيزون", ()))),
    Measure("ما متوسط عدد أيام صرف {X}{F}{G}؟", ("كم يوم بالمعدل ينصرف {GX}{F}{G}", "كم يوم يعطون {GX} بالمعدل{F}{G}"),
            ("sex", "age", "nat"), False, groups=("nat", "age", "year"),
            slots=(("الإنسولين|الانسولين", ()), ("الميتفورمين|الميتفورمين", ()), ("المضادات الحيوية|المضاد الحيوي", ()),
                   ("مكملات فيتامين د|مكملات فيتامين د", ()), ("أدوية الضغط|دوا الضغط", ()))),
    Measure("ما أكثر فئة دواء وُصفت{F}؟", ("شنو أكثر دوا انصرف{F}", "وش أكثر دوا يعطونه{F}", "أي دوا انصرف أكثر شي{F}"),
            ("sex", "age", "nat", "clinic", "dept"), False, groups=()),
    Measure("كم عدد {X}{F}{G}؟", ("كم {GX} صارت{F}{G}", "كم {GX} كانت{F}{G}"),
            ("sex", "age", "nat", "clinic", "dept", "weekday"), False, groups=("clinic", "dept", "nat", "age", "year", "month"),
            slots=(("زيارات المتابعة|زيارة متابعة", ()), ("زيارات الشكاوى الجديدة|زيارة شكوى جديدة", ()),
                   ("زيارات الفحص الدوري|زيارة فحص دوري", ()), ("زيارات التطعيم|زيارة تطعيم", ()))),
]

PREFIXES = [("من فضلك، ", "لو سمحت "), ("أريد أن أعرف ", "ابي اعرف "), ("أود أن أعرف ", "ودي اعرف "),
            ("أريد أن أعرف ", "ابغى اعرف "), ("أريد أن أعرف ", "بغيت اعرف "), ("أخبرني ", "قول لي "),
            ("دعنا نرى ", "خلنا نشوف "), ("", "يا ترى ")]


def _fill_slots(rng: random.Random, m: Measure, msa: str, gulf: str) -> tuple[str, str]:
    if not m.slots:
        return msa, gulf
    msa_spec, gulf_options = rng.choice(m.slots)
    name, *rest = msa_spec.split("|")
    msa = msa.replace("{X}", name)
    if len(rest) == 1:  # "MSA|Gulf"
        return msa, gulf.replace("{GX}", rest[0]).replace("{X}", rest[0])
    gulf = gulf.replace("{X}", rng.choice(gulf_options))
    if rest:
        msa, gulf = msa.replace("{V}", rest[0]), gulf.replace("{GV}", rest[1])
    return msa, gulf


def generate_one(rng: random.Random) -> tuple[str, str, str]:
    """One (gulf, msa, kind) question pair; kind is 'plain', 'grouped' or 'top'."""
    m = rng.choice(MEASURES)
    kind = rng.choices(["plain", "grouped", "top"], weights=[55, 30 if m.groups else 0, 15 if m.top and m.groups else 0])[0]

    group = rng.choice([g for g in m.groups if kind != "top" or GROUPS[g][2]]) if kind != "plain" else None
    dims = [d for d in m.dims if d != group]
    entity = [_entity_filter(rng, d, m.people) for d in rng.sample(dims, k=min(len(dims), rng.choice([0, 1, 1, 2])))]
    time = []
    if m.time and group not in ("year", "month") and rng.random() < 0.8:
        time = [_time_filters(rng, m.time)]
    elif m.time and group == "month":  # "per month" needs a year
        y = rng.choice(YEARS)
        time = [(f"في عام {y}", rng.choice([f"في {y}", f"سنة {y}"]))]

    msa_filters = "".join(f" {f[0]}" for f in entity + time)
    gulf_parts = entity + time
    rng.shuffle(gulf_parts)
    gulf_filters = "".join(f" {f[1]}" for f in gulf_parts)

    if kind == "top":
        _, singular, fih, _ = GROUPS[group]
        top_msa, top_gulf = rng.choice([m.top, m.low]) if m.low else m.top
        msa = f"أي {singular} كان {fih} {top_msa}{msa_filters}؟"
        gulf = f"أي {singular} {fih} {rng.choice(top_gulf)}{gulf_filters}"
    else:
        msa_group, gulf_group = _grouping(rng, group) if group else ("", "")
        msa = m.msa.replace("{F}", msa_filters).replace("{G}", f" {msa_group}" if group else "")
        gulf = rng.choice(m.gulf).replace("{F}", gulf_filters).replace("{G}", f" {gulf_group}" if group else "")
    msa, gulf = _fill_slots(rng, m, msa, gulf)

    if gulf.startswith("كم ") and rng.random() < 0.1:
        gulf = "چم " + gulf[3:]
    if rng.random() < 0.15:
        p_msa, p_gulf = rng.choice(PREFIXES)
        msa, gulf = p_msa + msa, p_gulf + gulf
    if rng.random() < 0.7:
        gulf += "؟"
    return " ".join(gulf.split()), " ".join(msa.split()), kind


def generate(n: int, seed: int = 13, identity_share: float = 0.1) -> list[dict]:
    """n distinct pairs; about `identity_share` of them map an MSA question to itself."""
    rng = random.Random(seed)
    seen, out = set(), []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        gulf, msa, kind = generate_one(rng)
        source = msa if rng.random() < identity_share else gulf
        if source in seen:
            continue
        seen.add(source)
        out.append({"id": f"clinic-{len(out)}", "source": source, "target": msa,
                    "kind": "identity" if source == msa else kind})
    return out
