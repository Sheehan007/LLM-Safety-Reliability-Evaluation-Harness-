# Architecture

The harness separates data generation, inference, scoring, and reporting so any layer can be replaced
without rewriting the rest of the pipeline.

```text
seed prompts ──> deterministic perturbations ──> JSONL benchmark
                                                    │
                         ┌──────────────────────────┴─────────────────────────┐
                         │                                                    │
              deterministic mock backend                         Hugging Face backend
                   (CI/validation)                              (PyTorch inference)
                         │                                                    │
                         └──────────────────────────┬─────────────────────────┘
                                                    v
                                      case-level metric records
                                                    │
                                                    v
                                  Pandas aggregation + failure reports
```

## Core modules

- `schemas.py` defines validated cases, model entries, and run configuration.
- `dataset.py` reads JSONL, detects duplicate IDs, expands seed cases, and creates the 500-case slice.
- `perturbations.py` implements deterministic, traceable prompt variations.
- `backends/` contains a shared interface, a CI-safe behavioral simulator, and batched Transformers
  inference.
- `metrics.py` computes case-level factuality, instruction, refusal, injection, similarity, and task
  success signals.
- `runner.py` performs batched execution and writes each completed batch immediately for recovery.
- `reporting.py` uses Pandas to produce model summaries, category tables, failure CSVs, and one Markdown
  report per model.

The backend receives structured evaluation cases rather than plain strings. That keeps deterministic
test doubles useful while allowing production backends to consume only the prompt.

