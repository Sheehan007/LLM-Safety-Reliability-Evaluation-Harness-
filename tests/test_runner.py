from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_safety_harness.dataset import expand_seed_cases, load_cases, write_jsonl
from llm_safety_harness.runner import run_evaluation
from llm_safety_harness.schemas import ModelConfig, RunConfig

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_end_to_end_mock_run_writes_reports(self) -> None:
        seeds = load_cases(ROOT / "data/seeds/seed_prompts.jsonl")[:3]
        cases = expand_seed_cases(seeds, variants_per_seed=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "prompts.jsonl"
            output = root / "findings"
            write_jsonl(dataset, (case.to_dict() for case in cases))
            config = RunConfig(
                run_name="test-run",
                result_kind="synthetic_validation",
                dataset_path=dataset,
                output_dir=output,
                models=(
                    ModelConfig(id="mock/a", backend="mock"),
                    ModelConfig(id="mock/b", backend="mock"),
                ),
                batch_size=3,
                resume=False,
            )

            result = run_evaluation(config)

            self.assertEqual(result["result_row_count"], 12)
            self.assertTrue((output / "report.md").exists())
            self.assertTrue((output / "model_summary.csv").exists())
            self.assertTrue((output / "stress_test_500_summary.csv").exists())
            self.assertEqual(len(list((output / "models").glob("*.md"))), 2)


if __name__ == "__main__":
    unittest.main()
