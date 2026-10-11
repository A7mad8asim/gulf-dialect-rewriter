"""End-to-end evaluation: ask-the-data's accuracy with and without a rewriter in front of it.

Runs with ask-the-data's Python (it imports `askdata`), from either repository:

    ..\\Ask-the-Data\\.venv\\Scripts\\python eval\\extrinsic.py --repeats 3

Arms: `none` (the question as asked) and `rewrite` (the question rewritten by the model, through
the guard; a rejected rewrite falls back to the original). Each question is rewritten once
(temperature 0) and both arms go through ask-the-data `repeats` times, because identical runs of
ask-the-data differ by a question or two. Results go to eval/results/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gdr.rewrite import OllamaRewriter  # noqa: E402

SUBSETS = {
    "dialect": lambda g: g["lang"] == "ar" and g["dialect"],
    "msa": lambda g: g["lang"] == "ar" and not g["dialect"],
    "arabic": lambda g: g["lang"] == "ar",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", choices=list(SUBSETS), default="dialect")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model that rewrites")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", default=None, help="name for the results file (default: the model)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    from askdata.config import Settings
    from askdata.db import Database
    from askdata.evaluation import _make_assistant, evaluate
    from askdata.resources import load_gold

    settings = Settings.from_env()
    gold = [g for g in load_gold() if SUBSETS[args.subset](g)]
    rewriter = OllamaRewriter(args.model, settings.ollama_base_url)

    print(f"Rewriting {len(gold)} {args.subset} questions with {rewriter.name}")
    rewrites = {}
    for g in gold:
        r = rewriter.rewrite(g["question"])
        rewrites[g["id"]] = r
        mark = "ok " if r.guard.accepted else "REJ"
        print(f"  {mark} {g['id']:<7} {g['question']}\n      -> {r.raw}" + ("" if r.guard.accepted else f"   [{'; '.join(r.guard.reasons)}]"))

    db = Database(settings.db_path, settings.query_timeout_s, settings.max_rows)
    assistant = _make_assistant(settings, "full", answers=False, db=db)
    arms = {"none": gold, "rewrite": [{**g, "question": rewrites[g["id"]].text} for g in gold]}
    correct: dict[str, dict[str, list[bool]]] = {arm: {g["id"]: [] for g in gold} for arm in arms}
    for rep in range(args.repeats):
        for arm, items in arms.items():
            print(f"\n== run {rep + 1}/{args.repeats}, arm: {arm} ==")
            report = evaluate(assistant, items, [], progress=True)
            for item in report["items"]:
                correct[arm][item["id"]].append(item["correct"])

    n = len(gold)
    summary = {}
    for arm in arms:
        per_run = [sum(correct[arm][g["id"]][k] for g in gold) for k in range(args.repeats)]
        summary[arm] = {"correct_per_run": per_run, "mean_accuracy": round(statistics.mean(per_run) / n, 4)}
    gained = [g["id"] for g in gold if sum(correct["rewrite"][g["id"]]) > sum(correct["none"][g["id"]])]
    lost = [g["id"] for g in gold if sum(correct["rewrite"][g["id"]]) < sum(correct["none"][g["id"]])]

    result = {
        "run": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "subset": args.subset,
            "n_questions": n,
            "repeats": args.repeats,
            "rewriter": rewriter.name,
            "text_to_sql": f"ask-the-data full pipeline, {settings.llm_provider}:{settings.llm_model}",
        },
        "summary": summary,
        "guard_rejected": sum(not r.guard.accepted for r in rewrites.values()),
        "rewrite_latency_median_s": round(statistics.median(r.latency_s for r in rewrites.values()), 2),
        "questions_better_with_rewrite": gained,
        "questions_worse_with_rewrite": lost,
        "items": [
            {
                "id": g["id"],
                "question": g["question"],
                "rewrite": rewrites[g["id"]].raw,
                "guard_accepted": rewrites[g["id"]].guard.accepted,
                "guard_reasons": rewrites[g["id"]].guard.reasons,
                "correct_none": correct["none"][g["id"]],
                "correct_rewrite": correct["rewrite"][g["id"]],
            }
            for g in gold
        ],
    }
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or args.model.replace(":", "-")
    path = out / f"extrinsic_{args.subset}_{label}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{args.subset}: {n} questions x {args.repeats} runs   (rewriter: {rewriter.name})")
    for arm, s in summary.items():
        print(f"  {arm:<8} {100 * s['mean_accuracy']:5.1f}%   correct per run: {s['correct_per_run']}")
    print(f"  guard rejected {result['guard_rejected']} rewrites; better with rewrite: {gained or '-'}; worse: {lost or '-'}")
    print(f"saved: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
