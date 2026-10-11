# Gulf-dialect rewriter

[![tests](https://github.com/A7mad8asim/gulf-dialect-rewriter/actions/workflows/tests.yml/badge.svg)](https://github.com/A7mad8asim/gulf-dialect-rewriter/actions/workflows/tests.yml)

A small open model, fine-tuned with LoRA/QLoRA, that rewrites Gulf-dialect Arabic questions into Modern Standard Arabic (MSA), placed in front of a text-to-SQL assistant to close its dialect gap.

> **Status: in progress.** No results yet. This README will lead with the measured before/after numbers once they exist.

## The question

[ask-the-data-text-to-sql](https://github.com/A7mad8asim/ask-the-data-text-to-sql) answers 80.6% of Modern Standard Arabic questions correctly but only 63.2% of Gulf-dialect ones (19 questions in its gold set). Does a fine-tuned dialect → MSA rewriter in front of it close that gap, without hurting the questions that were already right?

## Design rule

**The rewriter may change wording, never content.** Every number and Latin-script term (years, thresholds, HbA1c, BMI ...) must survive the rewrite unchanged, and none may be added. A rewrite that breaks the rule is discarded and the original question is used. This is enforced in code ([`guard.py`](src/gdr/guard.py)) and tested, not only asked for in a prompt.

## Plan

1. **Data.** Public Gulf-dialect ↔ MSA parallel text, fetched by a script and never committed. Every pair goes through the leakage check ([`leakage.py`](src/gdr/leakage.py)) against ask-the-data's gold set, so no test question (or a near-duplicate of one) reaches training.
2. **Baselines.** No rewrite, a prompted (not fine-tuned) base model, and the fine-tuned model, all through the same guard.
3. **Fine-tuning.** QLoRA on a small Qwen model on one RTX 5060 Ti (16 GB).
4. **Evaluation.**
   - Intrinsic: chrF / BLEU against reference MSA on a held-out split, plus the guard's rejection rate.
   - Extrinsic: ask-the-data's accuracy on its 19 Gulf-dialect questions before and after, and on the MSA questions to check nothing gets worse.
5. **Release.** Adapter weights and a model card on Hugging Face, a dataset card for the processed pairs (where the source licences allow), linked from here.

## Quick start (development)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Training needs the `train` extra (`pip install -e ".[train]"`) and a CUDA build of PyTorch.

Two checks work today:

```bash
gdr guard "كم مريض سكري عندنا فوق 60 سنة؟" "كم عدد مرضى السكري؟"
gdr check-leakage data/pairs.jsonl --held-out ../Ask-the-Data/eval/gold.jsonl --out data/pairs.clean.jsonl
```

## Limits

To be filled in with measured results.
