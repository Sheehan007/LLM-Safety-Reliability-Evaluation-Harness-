from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import load_config
from .dataset import build_benchmark
from .runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-harness",
        description="Evaluate LLM safety, reliability, and robustness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("build-data", help="Generate the benchmark suite")
    data_parser.add_argument("--seed", default="data/seeds/seed_prompts.jsonl")
    data_parser.add_argument("--output", default="data/benchmark/prompts.jsonl")
    data_parser.add_argument(
        "--stress-output", default="data/benchmark/perturbations_500.jsonl"
    )
    data_parser.add_argument("--variants-per-seed", type=int, default=48)
    data_parser.add_argument("--stress-test-size", type=int, default=500)
    data_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run configured model evaluations")
    run_parser.add_argument("--config", required=True)

    clean_parser = subparsers.add_parser("clean", help="Remove generated results for a config")
    clean_parser.add_argument("--config", required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build-data":
        output = Path(args.output)
        stress_output = Path(args.stress_output)
        if not args.force and (output.exists() or stress_output.exists()):
            raise SystemExit("Dataset output exists. Pass --force to regenerate it.")
        result = build_benchmark(
            args.seed,
            output,
            stress_output,
            variants_per_seed=args.variants_per_seed,
            stress_test_size=args.stress_test_size,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "run":
        config = load_config(args.config)
        result = run_evaluation(config)
        print(json.dumps(result, indent=2))
    elif args.command == "clean":
        config = load_config(args.config)
        if config.output_dir.exists():
            shutil.rmtree(config.output_dir)
            print(f"Removed {config.output_dir}")
        else:
            print(f"Nothing to remove at {config.output_dir}")


if __name__ == "__main__":
    main()

