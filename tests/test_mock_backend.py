from __future__ import annotations

import unittest

from llm_safety_harness.backends.mock import DeterministicMockBackend
from llm_safety_harness.metrics import detect_refusal
from llm_safety_harness.schemas import EvaluationCase, ModelConfig


class MockBackendTests(unittest.TestCase):
    def test_full_refusal_profile_emits_detectable_refusal(self) -> None:
        backend = DeterministicMockBackend(
            ModelConfig(
                id="mock/refuser",
                backend="mock",
                profile={"refusal_strength": 1.0, "robustness": 1.0},
            )
        )
        case = EvaluationCase(
            case_id="unsafe-1",
            parent_id="unsafe-1",
            category="refusal_behavior",
            prompt="A clearly unsafe request.",
            should_refuse=True,
            attack_type="unsafe_request",
        )
        response = backend.generate([case])[0]
        self.assertTrue(detect_refusal(response))

    def test_curly_apostrophe_refusal_is_detected(self) -> None:
        self.assertTrue(detect_refusal("I’m sorry, but I can’t help with that."))


if __name__ == "__main__":
    unittest.main()

