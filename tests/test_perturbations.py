from __future__ import annotations

import unittest

from llm_safety_harness.perturbations import PERTURBATION_FAMILIES, perturb_prompt


class PerturbationTests(unittest.TestCase):
    def test_original_is_unchanged(self) -> None:
        prompt, family, metadata = perturb_prompt("Hello world?", 0)
        self.assertEqual(prompt, "Hello world?")
        self.assertEqual(family, "original")
        self.assertEqual(metadata["perturbation_level"], 0)

    def test_variations_are_deterministic(self) -> None:
        first = perturb_prompt("Please calculate the answer.", 10)
        second = perturb_prompt("Please calculate the answer.", 10)
        self.assertEqual(first, second)

    def test_all_families_are_reachable(self) -> None:
        observed = {
            perturb_prompt("A sufficiently detailed prompt?", index)[1]
            for index in range(1, 12)
        }
        self.assertEqual(observed, set(PERTURBATION_FAMILIES))


if __name__ == "__main__":
    unittest.main()
