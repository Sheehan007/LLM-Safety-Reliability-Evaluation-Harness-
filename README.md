# LLM Safety & Reliability Evaluation Harness

A modular, reproducible framework for benchmarking factuality, instruction following, prompt
robustness, injection resistance, jailbreak resistance, and refusal behavior across local open-weight
language models.

The repository ships with a **1,536-prompt controlled evaluation suite**, a fixed **500-perturbation
stress test**, a production Hugging Face/PyTorch backend, and automated Pandas reports. A deterministic
six-profile validation run exercises the entire pipeline without downloading model weights; it is
clearly separated from empirical model findings.

## What is included

| Capability | Implementation |
| --- | --- |
| Multi-model inference | Batched `transformers` + PyTorch backend with chat-template support |
| Benchmark design | 32 curated seeds expanded into 1,536 traceable prompt versions |
| Robustness testing | 11 perturbation families and a committed 500-case stress-test slice |
| Safety signals | Attack-success rate, refusal rate, refusal consistency, false refusals |
| Reliability signals | Accuracy, instruction adherence, similarity, task success, behavioral drift |
| Failure analysis | Row-level CSV plus an automatically generated Markdown report per model |
| Reproducibility | Fixed seeds, dataset hash, environment manifest, deterministic decoding, resume support |
| Engineering quality | Typed package, unit/integration tests, lint configuration, GitHub Actions workflow |

## Repository map

```text
llm-harness/
├── configs/                 # Synthetic validation and six-model production runs
├── data/
│   ├── seeds/               # 32 curated source cases
│   └── benchmark/           # 1,536 prompts + the 500-perturbation slice
├── docs/                    # Architecture, methodology, setup, limitations, claim guide
├── findings/
│   ├── validation/          # Reproducible synthetic pipeline findings
│   └── open_source_6/       # Reserved for empirical six-model findings
├── src/llm_safety_harness/  # Backends, metrics, runner, data generation, reporting, CLI
└── tests/                   # Unit and end-to-end tests
```

## Quick start: validate the whole pipeline

Python 3.10–3.12 is recommended. This path does not download any model weights.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m llm_safety_harness build-data --force
python -m llm_safety_harness run --config configs/validation.json
python -m unittest discover -s tests -v
```

The validation evaluates six deterministic behavioral profiles over all 1,536 prompts, producing 9,216
model-prompt records. Its purpose is to verify scoring and reporting behavior—not to stand in for LLM
measurements. Read the generated [`findings/validation/report.md`](findings/validation/report.md) and
[`run_manifest.json`](findings/validation/run_manifest.json).

## Run the six open-source models

Install the optional ML stack and run the resumable configuration:

```bash
python -m pip install -e ".[ml]"
python -m llm_safety_harness run --config configs/open_source_6.json
```

The production config includes:

1. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
2. `Qwen/Qwen2.5-1.5B-Instruct`
3. `meta-llama/Llama-3.2-1B-Instruct`
4. `google/gemma-2-2b-it`
5. `microsoft/Phi-3.5-mini-instruct`
6. `mistralai/Mistral-7B-Instruct-v0.3`

Some weights are license-gated and require a Hugging Face account token. The largest entry also needs
substantially more memory than the 1–2B models. See [`docs/model_setup.md`](docs/model_setup.md) for a
low-memory workflow.

## Outputs

Each run directory is self-contained:

| Artifact | Purpose |
| --- | --- |
| `raw_results.jsonl` | Append-only checkpoint and complete response provenance |
| `scored_results.csv` | Case-level prompts, responses, metrics, drift, and failure labels |
| `model_summary.csv` | One comparative row per model |
| `category_metrics.csv` | Model × behavior breakdown |
| `stress_test_500_summary.csv` | Comparative metrics on the fixed 500-perturbation slice |
| `failures.csv` | Every flagged case with one or more failure reasons |
| `model_failure_summary.csv` | Failure taxonomy counts by model |
| `models/*.md` | Representative failures and category metrics for each model |
| `report.md` | Human-readable comparative findings |
| `run_manifest.json` | Result type, model IDs, counts, dataset hash, and library versions |

## Metric definitions

Attack-success rate and behavioral drift are **lower-is-better**. Accuracy, task success, instruction
adherence, and consistency are **higher-is-better**. Refusal rate is descriptive: a higher value may mean
better unsafe-request handling or excessive refusal of benign prompts.

The scorers favor transparency over hidden model judges. They combine normalized reference matching,
explicit format constraints, conservative refusal patterns, injected markers, and token cosine
similarity. Read [`docs/methodology.md`](docs/methodology.md) before interpreting a ranking.

## Reproduce the committed findings

```bash
./scripts/reproduce_validation.sh
```

That command regenerates the benchmark, reruns all six deterministic profiles, rebuilds every report,
and executes the tests. The dataset SHA-256 in the new manifest should match the committed run.

## Extending the harness

- Add prompt seeds as JSONL records under `data/seeds/` and regenerate the suite.
- Add a backend by implementing `ModelBackend.generate()` in `src/llm_safety_harness/backends/`.
- Add model entries to a copied JSON config; use a separate output directory per experiment.
- Add a new scorer in `metrics.py`, then expose it in the Pandas summaries and failure taxonomy.

## Responsible interpretation

This is an evaluation tool, not a safety certification. The included dataset is English-only,
single-turn, and intentionally compact at the seed level. Heuristic metrics can miss paraphrases or
novel refusal styles. Use human review and independently validated graders before making deployment
decisions. See [`docs/limitations.md`](docs/limitations.md).

## License

MIT
