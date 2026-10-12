"""Intrinsic evaluation: BLEU and chrF++ on the Dial2MSA-Verified Gulf test set (2,000 tweets, 3 MSA references each).

    .venv\\Scripts\\python eval\\intrinsic.py outputs/qwen3-1.7b-v2/adapter
    .venv\\Scripts\\python eval\\intrinsic.py --identity          # copy the tweet unchanged: the floor

Scored with sacrebleu against all three references, as Khered et al. (WACL 2025) report it, so the
numbers are comparable with their AraT5 results. Rewrites and scores go to eval/results/. Needs the
train and eval extras and `gdr fetch` (the test tweets are research-only and never committed; the
saved rewrites contain model output only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gdr.guard import check  # noqa: E402

TEST = ROOT / "data" / "pairs" / "test_dial2msa_glf.jsonl"
PUBLISHED = {  # Khered et al. 2025, Table 7 (AraT5, J-R configuration, three references)
    "AraT5 (Khered et al. 2025)": {"bleu": 53.96, "chrf++": 70.86},
}


def score(hypotheses: list[str], references: list[list[str]]) -> dict:
    import sacrebleu

    streams = [list(r) for r in zip(*references)]  # sacrebleu wants one stream per reference position
    return {
        "bleu": round(sacrebleu.corpus_bleu(hypotheses, streams).score, 2),
        "chrf++": round(sacrebleu.corpus_chrf(hypotheses, streams, word_order=2).score, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("adapter", nargs="?", help="outputs/<run>/adapter")
    parser.add_argument("--identity", action="store_true", help="score the unchanged tweets")
    parser.add_argument("--limit", type=int, help="first N tweets only (quick check)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if not TEST.exists():
        sys.exit(f"{TEST} is missing: run `gdr fetch` first")
    items = [json.loads(line) for line in TEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = items[: args.limit] if args.limit else items
    # every tweet has 3 references; pad defensively so the reference streams line up
    refs = [(i["references"] * 3)[:3] for i in items]

    if args.identity:
        name, hyps, raw, seconds = "identity", [i["source"] for i in items], None, 0.0
    else:
        from gdr.rewrite import AdapterRewriter

        rewriter = AdapterRewriter(args.adapter)
        name, raw, started = rewriter.name.split(":", 1)[1], [], time.time()
        for n, item in enumerate(items, 1):
            raw.append(rewriter.rewrite(item["source"]).raw)
            if n % 200 == 0:
                print(f"  {n}/{len(items)}  {time.time() - started:.0f}s", flush=True)
        seconds = time.time() - started
        hyps = raw

    result = {"rewriter": name, "n": len(items), **score(hyps, refs)}
    if raw is not None:
        guarded = [check(i["source"], r) for i, r in zip(items, raw)]
        result["guard_rejected"] = sum(not g.accepted for g in guarded)
        result["with_guard"] = score([g.text for g in guarded], refs)  # rejected rewrites fall back to the tweet
        result["seconds_per_tweet"] = round(seconds / len(items), 3)
    result["published"] = PUBLISHED

    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"intrinsic_{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if raw is not None:
        (out / f"intrinsic_rewrites_{name}.json").write_text(
            json.dumps([{"id": i["id"], "rewrite": r} for i, r in zip(items, raw)], ensure_ascii=False, indent=0),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
