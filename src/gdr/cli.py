"""Command line: gdr fetch | clinic | check-leakage | guard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .guard import check
from .leakage import LeakageChecker, load_questions

# ask-the-data's gold set, when the two projects sit side by side (Desktop\Ask-the-Data)
DEFAULT_HELD_OUT = Path(__file__).resolve().parents[3] / "Ask-the-Data" / "eval" / "gold.jsonl"


def _held_out(args: argparse.Namespace) -> list[str] | None:
    """ask-the-data's questions, [] with a warning when the default path is missing, None on a bad --held-out."""
    path = Path(args.held_out) if args.held_out else DEFAULT_HELD_OUT
    if path.exists():
        questions = load_questions(path)
        print(f"Held-out questions: {len(questions)} from {path}")
        return questions
    if args.held_out:
        print(f"Not found: {path}", file=sys.stderr)
        return None
    print(f"Warning: {path} not found; pairs are not checked against ask-the-data's questions")
    return []


def _fetch(args: argparse.Namespace) -> int:
    from .data import fetch

    held_out = _held_out(args)
    if held_out is None:
        return 1
    print(json.dumps(fetch(Path(args.data_dir), held_out, refresh=args.refresh), indent=2))
    return 0


def _clinic(args: argparse.Namespace) -> int:
    from .data import build_clinic

    held_out = _held_out(args)
    if held_out is None:
        return 1
    print(json.dumps(build_clinic(Path(args.data_dir), held_out, args.n, args.seed), indent=2, ensure_ascii=False))
    return 0


def _train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    overrides = {k: v for k, v in vars(args).items() if k not in ("command", "func") and v is not None}
    train(TrainConfig(**overrides))
    return 0


def _rewrite_file(args: argparse.Namespace) -> int:
    from .rewrite import AdapterRewriter, OllamaRewriter, rewrite_file

    rewriter = AdapterRewriter(args.adapter) if args.adapter else OllamaRewriter(args.ollama_model)
    rows = rewrite_file(rewriter, args.questions, args.out, dialect_only=args.dialect_only)
    print(f"{len(rows)} rewrites by {rewriter.name} -> {args.out}")
    return 0


def _check_leakage(args: argparse.Namespace) -> int:
    checker = LeakageChecker(load_questions(args.held_out))
    with open(args.pairs, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    kept, dropped = checker.filter_pairs(pairs)
    print(f"{len(pairs)} pairs: {len(kept)} kept, {len(dropped)} overlap the held-out questions")
    for p in dropped[: args.show]:
        print(f"  dropped: {p['source']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for p in kept:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return 0


def _guard(args: argparse.Namespace) -> int:
    result = check(args.question, args.rewrite)
    print(json.dumps({"text": result.text, "accepted": result.accepted, "reasons": result.reasons}, ensure_ascii=False))
    return 0 if result.accepted else 1


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="gdr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download the corpora and write train/dev/test pairs under data/pairs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--held-out", help=f"JSONL with a 'question' field (default: {DEFAULT_HELD_OUT})")
    p.add_argument("--refresh", action="store_true", help="download again even if cached")
    p.set_defaults(func=_fetch)

    p = sub.add_parser("clinic", help="generate clinic-question pairs into data/pairs/clinic_*.jsonl")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--held-out", help=f"JSONL with a 'question' field (default: {DEFAULT_HELD_OUT})")
    p.set_defaults(func=_clinic)

    p = sub.add_parser("train", help="QLoRA fine-tuning (needs the train extra and a CUDA GPU)")
    p.add_argument("--model", help="Hugging Face model id (default: Qwen/Qwen3-1.7B)")
    p.add_argument("--name", help="run name, the folder under outputs/")
    p.add_argument("--open-only", action="store_const", const=True, help="only openly licensed pairs (no Dial2MSA)")
    p.add_argument("--epochs", type=float)
    p.add_argument("--max-steps", type=int, help="stop after N steps (smoke test)")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lora-r", type=int)
    p.add_argument("--eval-every", type=int)
    p.set_defaults(func=_train)

    p = sub.add_parser("rewrite-file", help="rewrite every question of a JSONL file (for eval/extrinsic.py --rewrites)")
    p.add_argument("questions", help="JSONL with id and question (ask-the-data's gold set, eval/fresh_dialect.jsonl ...)")
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", help="outputs/<run>/adapter from gdr train")
    p.add_argument("--ollama-model", default="qwen3:8b", help="used when no --adapter is given")
    p.add_argument("--dialect-only", action="store_true", help="keep only Arabic questions marked as dialect")
    p.set_defaults(func=_rewrite_file)

    p = sub.add_parser("check-leakage", help="drop training pairs that overlap held-out questions")
    p.add_argument("pairs", help="training pairs, JSONL with 'source' and 'target'")
    p.add_argument("--held-out", required=True, help="JSONL with a 'question' field (e.g. ask-the-data's eval/gold.jsonl)")
    p.add_argument("--out", help="write the kept pairs here")
    p.add_argument("--show", type=int, default=10)
    p.set_defaults(func=_check_leakage)

    p = sub.add_parser("guard", help="check one rewrite against the design rule")
    p.add_argument("question")
    p.add_argument("rewrite")
    p.set_defaults(func=_guard)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
