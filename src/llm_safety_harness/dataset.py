from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .perturbations import perturb_prompt
from .schemas import EvaluationCase

DEFAULT_VARIANTS_PER_SEED = 48
DEFAULT_STRESS_TEST_SIZE = 500


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_cases(path: str | Path) -> list[EvaluationCase]:
    cases = [EvaluationCase.from_dict(record) for record in read_jsonl(path)]
    case_ids = [case.case_id for case in cases]
    duplicates = [key for key, count in Counter(case_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate case IDs: {duplicates[:5]}")
    return cases


def expand_seed_cases(
    seed_cases: Iterable[EvaluationCase], variants_per_seed: int = DEFAULT_VARIANTS_PER_SEED
) -> list[EvaluationCase]:
    if variants_per_seed < 1:
        raise ValueError("variants_per_seed must be positive")

    expanded: list[EvaluationCase] = []
    for seed in seed_cases:
        for variation_index in range(variants_per_seed):
            prompt, perturbation, perturbation_metadata = perturb_prompt(
                seed.prompt, variation_index
            )
            metadata = dict(seed.metadata)
            metadata.update(perturbation_metadata)
            expanded.append(
                EvaluationCase(
                    case_id=f"{seed.case_id}--v{variation_index:02d}",
                    parent_id=seed.case_id,
                    category=seed.category,
                    prompt=prompt,
                    reference_answer=seed.reference_answer,
                    should_refuse=seed.should_refuse,
                    attack_type=seed.attack_type,
                    attack_marker=seed.attack_marker,
                    constraints=seed.constraints,
                    perturbation=perturbation,
                    variation_index=variation_index,
                    metadata=metadata,
                )
            )
    return expanded


def build_benchmark(
    seed_path: str | Path,
    benchmark_path: str | Path,
    stress_path: str | Path,
    *,
    variants_per_seed: int = DEFAULT_VARIANTS_PER_SEED,
    stress_test_size: int = DEFAULT_STRESS_TEST_SIZE,
) -> dict[str, object]:
    seeds = [EvaluationCase.from_dict(record) for record in read_jsonl(seed_path)]
    expanded = expand_seed_cases(seeds, variants_per_seed)
    perturbed = [case for case in expanded if case.variation_index > 0]
    if len(perturbed) < stress_test_size:
        raise ValueError("Not enough perturbed cases to construct the requested stress test")

    # Stable hash ordering spreads the fixed slice across categories, parents, and families.
    stride_order = sorted(
        perturbed,
        key=lambda case: hashlib.sha256(case.case_id.encode()).hexdigest(),
    )
    stress_cases = stride_order[:stress_test_size]
    stress_ids = {case.case_id for case in stress_cases}
    benchmark_records = [_with_stress_label(case, case.case_id in stress_ids) for case in expanded]
    stress_records = [_with_stress_label(case, True) for case in stress_cases]
    write_jsonl(benchmark_path, benchmark_records)
    write_jsonl(stress_path, stress_records)
    return {
        "seed_count": len(seeds),
        "benchmark_count": len(expanded),
        "stress_test_count": len(stress_cases),
        "category_counts": dict(Counter(case.category for case in expanded)),
        "variants_per_seed": variants_per_seed,
    }


def _with_stress_label(case: EvaluationCase, is_member: bool) -> dict:
    record = case.to_dict()
    record["metadata"]["stress_test_500"] = is_member
    return record
