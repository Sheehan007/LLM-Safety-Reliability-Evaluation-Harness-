from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..schemas import EvaluationCase, ModelConfig


class ModelBackend(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, cases: Sequence[EvaluationCase]) -> list[str]:
        """Generate one response for every case, preserving input order."""

    def close(self) -> None:
        """Release backend resources."""
        return None


def create_backend(config: ModelConfig) -> ModelBackend:
    if config.backend == "mock":
        from .mock import DeterministicMockBackend

        return DeterministicMockBackend(config)
    if config.backend == "huggingface":
        from .huggingface import HuggingFaceBackend

        return HuggingFaceBackend(config)
    raise ValueError(f"Unknown backend: {config.backend}")
