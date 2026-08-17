from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from llm_safety_harness.dataset import build_benchmark, load_cases
from llm_safety_harness.perturbations import PERTURBATION_FAMILIES

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_full_benchmark_has_expected_scale_and_categories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            benchmark = Path(directory) / "prompts.jsonl"
            stress = Path(directory) / "stress.jsonl"
            result = build_benchmark(
                ROOT / "data/seeds/seed_prompts.jsonl", benchmark, stress
            )

            self.assertEqual(result["seed_count"], 32)
            self.assertEqual(result["benchmark_count"], 1536)
            self.assertEqual(result["stress_test_count"], 500)
            cases = load_cases(benchmark)
            counts = Counter(case.category for case in cases)
            self.assertEqual(set(counts), {
                "factuality",
                "instruction_following",
                "robustness",
                "prompt_injection",
                "jailbreak",
                "refusal_behavior",
            })
            stress_cases = load_cases(stress)
            self.assertEqual(
                {case.perturbation for case in stress_cases}, set(PERTURBATION_FAMILIES)
            )
            self.assertEqual(len({case.parent_id for case in stress_cases}), 32)
            self.assertEqual(
                sum(bool(case.metadata.get("stress_test_500")) for case in cases), 500
            )
            self.assertTrue(
                all(case.metadata.get("stress_test_500") for case in stress_cases)
            )

    def test_generated_case_ids_are_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            benchmark = Path(directory) / "prompts.jsonl"
            stress = Path(directory) / "stress.jsonl"
            build_benchmark(
                ROOT / "data/seeds/seed_prompts.jsonl",
                benchmark,
                stress,
                variants_per_seed=2,
                stress_test_size=16,
            )
            cases = load_cases(benchmark)
            self.assertEqual(len(cases), len({case.case_id for case in cases}))


if __name__ == "__main__":
    unittest.main()
