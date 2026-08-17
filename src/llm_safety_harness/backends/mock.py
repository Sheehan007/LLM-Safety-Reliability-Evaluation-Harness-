from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ..schemas import EvaluationCase, ModelConfig
from .base import ModelBackend

DEFAULT_PROFILE = {
    "base_accuracy": 0.85,
    "instruction_adherence": 0.85,
    "refusal_strength": 0.85,
    "injection_resistance": 0.85,
    "robustness": 0.85,
}


class DeterministicMockBackend(ModelBackend):
    """A repeatable behavioral simulator used only to validate harness plumbing.

    It never pretends to be a language model. Profiles create predictable strengths
    and failure modes so reports, metrics, resumption, and CI can be tested cheaply.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.profile = {**DEFAULT_PROFILE, **config.profile}
        for name, value in self.profile.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Mock profile value {name} must be between 0 and 1")

    def generate(self, cases: Sequence[EvaluationCase]) -> list[str]:
        return [self._generate_one(case) for case in cases]

    def _generate_one(self, case: EvaluationCase) -> str:
        perturbation_penalty = self._perturbation_penalty(case)

        if case.attack_type == "prompt_injection":
            resistance = self.profile["injection_resistance"] * perturbation_penalty
            if self._roll(case, "injection") <= resistance:
                return case.reference_answer or "I ignored the untrusted instruction."
            return case.attack_marker or "INJECTION_SUCCEEDED"

        if case.should_refuse:
            refusal_probability = self.profile["refusal_strength"] * perturbation_penalty
            if self._roll(case, "refusal") <= refusal_probability:
                return "I'm sorry, but I can't help with that request."
            return (
                "UNSAFE_COMPLIANCE: simulated policy failure; actionable details are "
                "intentionally omitted."
            )

        over_refusal_probability = max(0.0, self.profile["refusal_strength"] - 0.84) * 0.16
        if self._roll(case, "over-refusal") < over_refusal_probability:
            return "I'm sorry, but I can't help with that request."

        quality = self.profile["base_accuracy"]
        if case.category == "instruction_following":
            quality = self.profile["instruction_adherence"]
        quality *= perturbation_penalty
        if self._roll(case, "quality") <= quality:
            return case.reference_answer or "Completed."
        return "I cannot determine that from the information provided."

    def _perturbation_penalty(self, case: EvaluationCase) -> float:
        if case.variation_index == 0:
            return 1.0
        level = int(case.metadata.get("perturbation_level", 1))
        severity = min(1.0, 0.2 + level * 0.16)
        robustness = self.profile["robustness"]
        return max(0.05, 1.0 - (1.0 - robustness) * severity)

    def _roll(self, case: EvaluationCase, purpose: str) -> float:
        key = f"{self.config.id}|{case.case_id}|{purpose}".encode()
        integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        return integer / (2**64 - 1)
