#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH=src python -m llm_safety_harness build-data --force
PYTHONPATH=src python -m llm_safety_harness run --config configs/validation.json
PYTHONPATH=src python -m unittest discover -s tests -v

