from __future__ import annotations

import json
from pathlib import Path

from .schemas import ModelConfig, RunConfig


def load_config(path: str | Path, *, project_root: str | Path | None = None) -> RunConfig:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    models = tuple(ModelConfig.from_dict(item) for item in raw["models"])
    dataset_path = _resolve_path(raw["dataset_path"], root)
    output_dir = _resolve_path(raw["output_dir"], root)
    return RunConfig(
        run_name=raw["run_name"],
        result_kind=raw["result_kind"],
        dataset_path=dataset_path,
        output_dir=output_dir,
        models=models,
        batch_size=int(raw.get("batch_size", 4)),
        seed=int(raw.get("seed", 42)),
        resume=bool(raw.get("resume", True)),
    )


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

