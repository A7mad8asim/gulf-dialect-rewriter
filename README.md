# Gulf-dialect rewriter

[![tests](https://github.com/A7mad8asim/gulf-dialect-rewriter/actions/workflows/tests.yml/badge.svg)](https://github.com/A7mad8asim/gulf-dialect-rewriter/actions/workflows/tests.yml)

A small open model, fine-tuned with LoRA/QLoRA, that rewrites Gulf-dialect Arabic questions into Modern Standard Arabic (MSA), placed in front of a text-to-SQL assistant to close its dialect gap.

> **Status: in progress.** No results yet. This README will lead with the measured before/after numbers once they exist.

## The question

[ask-the-data-text-to-sql](https://github.com/A7mad8asim/ask-the-data-text-to-sql) answers 80.6% of Modern Standard Arabic questions correctly but only 63.2% of Gulf-dialect ones (19 questions in its gold set). Does a fine-tuned dialect → MSA rewriter in front of it close that gap, without hurting the questions that were already right?

## Design rule

**The rewriter may change wording, never content.** Every number and Latin-script term (years, thresholds, HbA1c, BMI ...) must survive the rewrite unchanged, and none may be added. A rewrite that breaks the rule is discarded and the original question is used. This is enforced in code ([`guard.py`](src/gdr/guard.py)) and tested, not only asked for in a prompt.

## Plan

1. **Data.** Public Gulf-dialect ↔ MSA parallel text (see [Data](#data)), fetched by `gdr fetch` and never committed, plus clinic-domain pairs from a seeded generator (`gdr clinic`). Every pair goes through the leakage check ([`leakage.py`](src/gdr/leakage.py)) against ask-the-data's gold set, so no test question (or a near-duplicate of one) reaches training.
2. **Baselines.** No rewrite, a prompted (not fine-tuned) base model, a hand-written MSA reference as the ceiling, and the fine-tuned model, all through the same guard. Done so far: [ask-the-data's 19](#baseline-prompted-qwen3-8b-not-fine-tuned) and the [fresh 25](#fresh-dialect-test-set).
3. **Fine-tuning.** QLoRA on a small Qwen model on one RTX 5060 Ti (16 GB).
4. **Evaluation.**
   - Intrinsic: chrF / BLEU against reference MSA on a held-out split, plus the guard's rejection rate.
   - Extrinsic: ask-the-data's accuracy on its 19 Gulf-dialect questions before and after, and on the MSA questions to check nothing gets worse.
5. **Release.** Adapter weights and a model card on Hugging Face, a dataset card for the processed pairs (where the source licences allow), linked from here.

## First fine-tuned model: Qwen3-1.7B, QLoRA

`gdr train` defaults: 4-bit NF4 base, LoRA r=16 on all linear layers (17.4 M trainable parameters, 1% of the model), 2 epochs over 11,816 pairs (Dial2MSA + STS Saudi + clinic). Training took 47 minutes on the RTX 5060 Ti, peaking at 11.3 GB. Dev loss fell from 0.48 to 0.385 and flattened over the last epoch. The rewrites come from `gdr rewrite-file` and are scored with `eval/extrinsic.py --rewrites`, 3 runs per arm, on 12 Oct 2026:

| Test set | No rewrite | Prompted Qwen3-8B | **Fine-tuned Qwen3-1.7B** | MSA reference |
|---|---|---|---|---|
| ask-the-data's 19 dialect questions | 57.9% | 57.9% | **73.7%** (+3, none lost) | n/a |
| Fresh 25 dialect questions | 44.0% | 48.0% | **40.0%** (+4, −5) | 72–76% |

**The fine-tuned model fixes clinic questions it was trained on, but it doesn't generalize yet.**

- **ask-the-data's 19 questions:** it fixes 3 of the 6 that a better rewrite could fix (G12, G14, G28) and breaks none. It also corrects the prompted model's G26 mistake: مريض ضغط → مرضى ارتفاع ضغط الدم. These questions are close to what the clinic generator produces, though: the generator draws on the same glossary, and Claude had seen the questions before writing it. So this result is optimistic.
- **The fresh 25:** questions outside the generator's templates bring out confident meaning changes. Sunday became "weekend" (F01), asthma became "lung cancer" (F03), and "days of insulin supplied" became "days between visits" (F06). F05, F16 and F18 came out as nonsense. The guard rejected none of them, because no number or term changed. Net −1 is within noise, but the meaning changes are real.
- **Noise:** the reference arm scored 18 one day and 19 the next with identical text, so judge arms on the per-question tables in `eval/results/` as well as on totals.

What to try next: more varied clinic training questions (more measures, filters and wordings; done on 12 Oct, see [Data](#data)), the 8B base, which knows more Arabic, and a check that catches meaning changes and not only lost numbers.

## Baseline: prompted Qwen3-8B, not fine-tuned

Ask-the-Data's 19 Gulf-dialect questions, each run 3 times through its full pipeline (Qwen3-8B writes the SQL). In the rewrite arm, the same Qwen3-8B first rewrites each question into MSA with a fixed prompt ([`rewrite.py`](src/gdr/rewrite.py)), behind the guard. Measured on 11 Oct 2026 with [`eval/extrinsic.py`](eval/extrinsic.py):

| Arm | Correct (each of 3 runs) | Accuracy |
|---|---|---|
| No rewrite | 11 · 11 · 11 | 57.9% |
| Prompted Qwen3-8B rewrite | 11 · 11 · 11 | 57.9% |

The rewrite fixes 2 questions and breaks 2, so the net effect is zero. The guard rejected none of the 19 rewrites, and the median rewrite takes 0.38 s.

- **Fixed:** G12 (انحجز فيها أكثر شي → أكبر عدد من المواعيد) and G28 (انحجز عن طريق التطبيق).
- **Broken:** G26 changes the meaning. مريض ضغط ("hypertension patient") became المرضى الذين تم تسجيل ضغطهم ("patients whose blood pressure was recorded"). The guard can't catch this, because no number or term changed. G31's rewrite is faithful (كم معدل → ما هو معدل), but the SQL model then got it wrong.
- **Room to improve:** 8 questions fail without a rewrite. The English twins of G06 and G45 fail as well, so dialect isn't their problem. That leaves 6 failures a better rewriter could fix: G12, G14, G28, G36, G39, G48. The prompted model fixed 2 of them.

The no-rewrite arm scores 57.9% (11 of 19), against the 63.2% (12 of 19) in Ask-the-Data's README from 7 Oct. G28 was right then and is wrong now. The 3 runs here were identical, but single questions still move between days, so compare arms on paired runs, as here.

## Fresh dialect test set

[`eval/fresh_dialect.jsonl`](eval/fresh_dialect.jsonl) holds 25 new Gulf-dialect clinic questions. Each has gold SQL checked against ask-the-data's database and a hand-written MSA reference. **Claude wrote them, not a native speaker**, after designing the generator and seeing ask-the-data's gold questions. That makes it a weaker check than an independent set, though it was built to differ from both:
- **Other question types:** weekday, time of day, asthma, LDL, overdue follow-ups, winter antibiotics, first diagnosis, labs per clinic, two-clinic comparisons.
- **Other Gulf wordings:** چم, شقد, ابغى, خلنا نشوف, اللي راحت, ياهل, تلتزم بمواعيدها.

A test keeps every question away from the generator's output and away from ask-the-data's gold questions. Each "which ..." question was checked to have a clear winner, not a near-tie.

Results on 11 Oct 2026, 3 runs per arm:

| Arm | Correct (each of 3 runs) | Accuracy |
|---|---|---|
| No rewrite | 11 · 11 · 11 | 44.0% |
| Prompted Qwen3-8B rewrite | 12 · 12 · 12 | 48.0% |
| Hand-written MSA reference (a perfect rewriter) | 18 · 18 · 18 | **72.0%** |

- **Ceiling:** a perfect rewrite into MSA would lift ask-the-data from 44% to 72% on these questions. That gap is what fine-tuning aims at.
- **The prompted model often changes the meaning:** كنسلوا (cancelled) became لم يحضروا (did not show up) in F13. دوا انصرف للعيال ("medicine prescribed to children") became "the highest-revenue amount for families" in F05. كم وحدة من الحريم ("how many women") became "how many units of women" in F04. F02 and F08 came back still in dialect. The guard rejected none of these, because no number or term changed. Its checks stop numbers from changing but can't catch a change in meaning.
- **Noise:** F08's rewrite is character-for-character the original question. Yet it failed in all 3 no-rewrite runs and passed in all 3 rewrite runs. Each arm repeats perfectly, but the same text can still flip between arms, so a difference of 1–2 questions is noise. The +1 for the prompted model is within it. The +7 for the reference is not.

## Data

`gdr fetch` downloads the corpora into `data/raw` (never committed) and writes JSONL pairs to `data/pairs`:

| Source | Pairs kept | Licence | Use |
|---|---|---|---|
| [Dial2MSA-Verified](https://github.com/khered20/Dial2MSA-Verified), Gulf portion (Khered et al., WACL 2025): tweets with human-verified MSA | 6,018 train · 199 dev · 2,000 test tweets with 3 references each | Research use only; not redistributed | Train, dev, intrinsic test |
| [Arabic STS](https://pmc.ncbi.nlm.nih.gov/articles/PMC9371328/), Saudi translations (Al Sulaiman et al., PLOS ONE 2022, file S1): MSA sentences translated into Saudi dialect by people | 3,124 train | CC BY 4.0 | Train |
| Clinic questions generated by `gdr clinic` (this repository) | 5,349 train · 594 dev | Own data | Train, dev |

Each pair records its licence (`open_licence: true` for CC BY), so an adapter can also be trained on the openly licensed pairs alone.

Training and dev pairs are filtered before use (`data/pairs/rejected.jsonl` lists every dropped pair with its reason):

- **Guard (155 dropped):** a reference that loses or adds a number or Latin term, mostly tweets with list numbers ("1 … 2 …") that the translators left out. Training on them would teach the model to break the design rule.
- **Overlap (415 dropped, 220 distinct tweets):** Dial2MSA training tweets that repeat or nearly repeat one of its own test tweets (20 exact copies, the rest retweets and small variants). None overlapped ask-the-data's gold questions.

**Why clinic questions are generated:** the tweets are social media, and their questions are mostly اشلون / وش (كم appears 18 times). ask-the-data's dialect questions are كم questions about appointments and lab results. [`clinic.py`](src/gdr/clinic.py) composes a question from four parts:

- **A measure (33):** appointments, attendance and no-show rates, cancellations, visits by type, patients seen and registered, waiting and lead time, chronic-condition patients and first diagnoses, vitamin D, LDL and HbA1c results, screening counts and abnormal rates, overdue and on-time follow-ups, prescriptions and days supplied for 8 drug classes, the most-prescribed drug.
- **Filters:** year, month or month range, quarter, Ramadan, summer, winter, Eid, weekend, weekday, appointment time, clinic, department, booking channel, sex, age, nationality.
- **A grouping, or a superlative** (most or fewest).
- **A Gulf opener (optional):** چم, ابغى اعرف, قول لي, خلنا نشوف, يا ترى ...

Each part has one MSA wording and several Gulf ones (تكنسل, انحجز, ما جوا, انصرف له, طلع, سوّى, انسوى, تسوت, الرياييل, الحريم, اليهال, الشيبان, الشياب, أكثر شي, بروحها, السنة اللي راحت ...). About 10% of pairs are MSA → MSA, so the rewriter learns to leave MSA alone. The clinic dev set shares templates with training, so it only checks fitting.

**Expansion on 12 Oct:** the generator went from 18 to 33 measures after the first fine-tuned model learned its templates. The new parts come from ask-the-data's schema and glossary, table by table, but they now cover the same areas as the fresh test set. `gdr clinic` therefore holds out the fresh 25 as well as ask-the-data's gold questions: 57 of 6,000 generated questions were dropped as too close to one of them. Even so, the fresh set is no longer fully independent of the training data. A final claim needs a new held-out set, written without looking at the generator.

Some tweets labelled Gulf are Iraqi or Kuwaiti.

## Quick start (development)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Training needs a CUDA build of PyTorch and the `train` extra. Tested on Windows 11 with an RTX 5060 Ti 16 GB (Blackwell, which needs CUDA 12.8 or later), torch 2.11.0+cu128, transformers 5.19, peft 0.21 and bitsandbytes 0.50 (4-bit NF4 with bf16 compute works). Training uses the plain transformers `Trainer` with its own masking ([`train.py`](src/gdr/train.py)). TRL's SFT trainer needs Triton for its fused loss, and Triton ships with PyTorch on Linux only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train]"
gdr train --max-steps 20 --name smoke   # check masking, losses, saving and sample rewrites in about 2 minutes
gdr train                               # Qwen3-1.7B, 4-bit QLoRA, all training pairs, 2 epochs
gdr train --open-only --name qwen3-1.7b-open   # openly licensed pairs only
```

```bash
pip install -e ".[data]"
gdr fetch                     # about 4 MB; checks against ../Ask-the-Data/eval/gold.jsonl when present
gdr clinic --n 6000 --seed 13 # clinic-question pairs, same filters
gdr guard "كم مريض سكري عندنا فوق 60 سنة؟" "كم عدد مرضى السكري؟"
gdr check-leakage data/pairs.jsonl --held-out ../Ask-the-Data/eval/gold.jsonl --out data/pairs.clean.jsonl
```

## Limits

To be filled in with measured results.
