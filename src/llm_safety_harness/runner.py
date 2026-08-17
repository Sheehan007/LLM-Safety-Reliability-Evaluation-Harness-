from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .backends import create_backend
from .dataset import load_cases, read_jsonl
from .metrics import score_case
from .reporting import build_reports
from .schemas import EvaluationCase, RunConfig


def batched(values: list[EvaluationCase], size: int) -> Iterable[list[EvaluationCase]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_evaluation(config: RunConfig) -> dict[str, Any]:
    cases = load_cases(config.dataset_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = config.output_dir / "raw_results.jsonl"
    completed = _completed_keys(raw_path) if config.resume else set()
    if raw_path.exists() and not config.resume:
        raw_path.unlink()

    _set_seed(config.seed)
    run_started = time.time()
    executed = 0

    for model_config in config.models:
        pending = [
            case for case in cases if (model_config.id, case.case_id) not in completed
        ]
        if not pending:
            continue
        backend = create_backend(model_config)
        try:
            for case_batch in batched(pending, config.batch_size):
                batch_started = time.perf_counter()
                responses = backend.generate(case_batch)
                elapsed_ms = (time.perf_counter() - batch_started) * 1000
                if len(responses) != len(case_batch):
                    raise RuntimeError(
                        f"Backend returned {len(responses)} responses for "
                        f"{len(case_batch)} prompts"
                    )
                latency_per_prompt = elapsed_ms / max(1, len(case_batch))
                rows = [
                    _result_row(
                        config=config,
                        model_id=model_config.id,
                        backend_name=model_config.backend,
                        case=case,
                        response=response,
                        latency_ms=latency_per_prompt,
                    )
                    for case, response in zip(case_batch, responses, strict=True)
                ]
                _append_jsonl(raw_path, rows)
                executed += len(rows)
        finally:
            backend.close()

    records = read_jsonl(raw_path)
    manifest = {
        "run_name": config.run_name,
        "result_kind": config.result_kind,
        "synthetic_results": config.result_kind == "synthetic_validation",
        "model_count": len(config.models),
        "models": [model.id for model in config.models],
        "dataset_path": _display_path(config.dataset_path),
        "dataset_sha256": _sha256_file(config.dataset_path),
        "dataset_prompt_count": len(cases),
        "stress_test_prompt_count": sum(
            bool(case.metadata.get("stress_test_500")) for case in cases
        ),
        "result_row_count": len(records),
        "executed_in_this_invocation": executed,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "elapsed_seconds": round(time.time() - run_started, 3),
        "python_version": platform.python_version(),
        "library_versions": _library_versions(),
        "methodology": "docs/methodology.md",
    }
    reports = build_reports(records, config.output_dir, manifest)
    return {**manifest, **reports}


def _result_row(
    *,
    config: RunConfig,
    model_id: str,
    backend_name: str,
    case: EvaluationCase,
    response: str,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "result_kind": config.result_kind,
        "model_id": model_id,
        "backend": backend_name,
        **case.to_dict(),
        "stress_test_member": bool(case.metadata.get("stress_test_500")),
        "response": response.strip(),
        "latency_ms": round(latency_ms, 4),
        **score_case(case, response),
    }


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(row["model_id"], row["case_id"]) for row in read_jsonl(path)}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("pandas", "torch", "transformers"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package] = None
    return versions


def config_as_dict(config: RunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["dataset_path"] = str(config.dataset_path)
    payload["output_dir"] = str(config.output_dir)
    return payload
