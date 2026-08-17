from __future__ import annotations

import unittest

from llm_safety_harness.metrics import (
    detect_refusal,
    instruction_adherence,
    semantic_similarity,
)


class MetricTests(unittest.TestCase):
    def test_refusal_detection(self) -> None:
        self.assertTrue(detect_refusal("I'm sorry, but I can't help with that request."))
        self.assertFalse(detect_refusal("Paris is the capital of France."))

    def test_instruction_constraints(self) -> None:
        constraints = {"required_prefix": "Summary:", "required_terms": ["light"]}
        self.assertEqual(
            instruction_adherence("Summary: plants use light.", constraints), 1.0
        )
        self.assertEqual(instruction_adherence("Plants use light.", constraints), 0.0)

    def test_json_constraint(self) -> None:
        self.assertEqual(instruction_adherence('{"status":"ok"}', {"format": "json"}), 1.0)
        self.assertEqual(instruction_adherence("status: ok", {"format": "json"}), 0.0)

    def test_semantic_similarity_bounds(self) -> None:
        self.assertAlmostEqual(semantic_similarity("alpha beta", "alpha beta") or 0, 1.0)
        different = semantic_similarity("alpha", "omega")
        self.assertIsNotNone(different)
        self.assertGreaterEqual(different or 0, 0.0)
        self.assertLessEqual(different or 0, 1.0)


if __name__ == "__main__":
    unittest.main()

