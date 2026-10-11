# Gulf-dialect rewriter

[![tests](https://github.com/A7mad8asim/gulf-dialect-rewriter/actions/workflows/tests.yml/badge.svg)](https://github.com/A7mad8asim/gulf-dialect-rewriter/actions/workflows/tests.yml)

A small open model, fine-tuned with LoRA/QLoRA, that rewrites Gulf-dialect Arabic questions into Modern Standard Arabic (MSA), placed in front of a text-to-SQL assistant to close its dialect gap.

> **Status: in progress.** No results yet. This README will lead with the measured before/after numbers once they exist.

## The question

[ask-the-data-text-to-sql](https://github.com/A7mad8asim/ask-the-data-text-to-sql) answers 80.6% of Modern Standard Arabic questions correctly but only 63.2% of Gulf-dialect ones (19 questions in its gold set). Does a fine-tuned dialect → MSA rewriter in front of it close that gap, without hurting the questions that were already right?

## Design rule

**The rewriter may change wording, never content.** Every number and Latin-script term (years, thresholds, HbA1c, BMI ...) must survive the rewrite unchanged, and none may be added. A rewrite that breaks the rule is discarded and the original question is used. This is enforced in code ([`guard.py`](src/gdr/guard.py)) and tested, not only asked for in a prompt.

## Plan

1. **Data.** Public Gulf-dialect ↔ MSA parallel text (see [Data](#data)), fetched by `gdr fetch` and never committed, plus clinic-domain pairs from a seeded generator (to do). Every pair goes through the leakage check ([`leakage.py`](src/gdr/leakage.py)) against ask-the-data's gold set, so no test question (or a near-duplicate of one) reaches training.
2. **Baselines.** No rewrite, a prompted (not fine-tuned) base model, and the fine-tuned model, all through the same guard.
3. **Fine-tuning.** QLoRA on a small Qwen model on one RTX 5060 Ti (16 GB).
4. **Evaluation.**
   - Intrinsic: chrF / BLEU against reference MSA on a held-out split, plus the guard's rejection rate.
   - Extrinsic: ask-the-data's accuracy on its 19 Gulf-dialect questions before and after, and on the MSA questions to check nothing gets worse.
5. **Release.** Adapter weights and a model card on Hugging Face, a dataset card for the processed pairs (where the source licences allow), linked from here.

## Data

`gdr fetch` downloads the corpora into `data/raw` (never committed) and writes JSONL pairs to `data/pairs`:

| Source | Pairs kept | Licence | Use |
|---|---|---|---|
| [Dial2MSA-Verified](https://github.com/khered20/Dial2MSA-Verified), Gulf portion (Khered et al., WACL 2025): tweets with human-verified MSA | 6,017 train · 199 dev · 2,000 test tweets with 3 references each | Research use only; not redistributed | Train, dev, intrinsic test |
| [Arabic STS](https://pmc.ncbi.nlm.nih.gov/articles/PMC9371328/), Saudi translations (Al Sulaiman et al., PLOS ONE 2022, file S1): MSA sentences translated into Saudi dialect by people | 3,118 train | CC BY 4.0 | Train |

Each pair records its licence (`open_licence: true` for CC BY), so an adapter can also be trained on the openly licensed pairs alone.

Training and dev pairs are filtered before use (`data/pairs/rejected.jsonl` lists every dropped pair with its reason):

- **Guard (162 dropped):** a reference that loses or adds a number or Latin term, mostly tweets with list numbers ("1 … 2 …") that the translators left out. Training on them would teach the model to break the design rule.
- **Overlap (415 dropped, 220 distinct tweets):** Dial2MSA training tweets that repeat or nearly repeat one of its own test tweets (20 exact copies, the rest retweets and small variants). None overlapped ask-the-data's gold questions.

Gaps: the tweets are social media, and their questions are mostly اشلون / وش. ask-the-data's dialect questions are كم questions about appointments and lab results (عندنا, انحجز, تكنسل, طلع, حريم, أكثر شي). Clinic-domain pairs from a seeded generator are the next step. Some tweets labelled Gulf are Iraqi or Kuwaiti.

## Quick start (development)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Training needs the `train` extra (`pip install -e ".[train]"`) and a CUDA build of PyTorch.

```bash
pip install -e ".[data]"
gdr fetch                     # about 4 MB; checks against ../Ask-the-Data/eval/gold.jsonl when present
gdr guard "كم مريض سكري عندنا فوق 60 سنة؟" "كم عدد مرضى السكري؟"
gdr check-leakage data/pairs.jsonl --held-out ../Ask-the-Data/eval/gold.jsonl --out data/pairs.clean.jsonl
```

## Limits

To be filled in with measured results.
