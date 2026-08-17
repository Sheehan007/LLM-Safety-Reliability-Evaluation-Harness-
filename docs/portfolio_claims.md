# Portfolio and résumé claim guide

The repository is designed to make every public claim traceable to an artifact.

## Claims supported immediately

- Architected a modular Python evaluation framework covering factuality, instruction following,
  robustness, prompt injection, jailbreak resistance, and refusal behavior.
- Built a deterministic 1,536-prompt suite with a fixed 500-perturbation stress test and automated
  model-level failure reports.
- Added batched PyTorch/Hugging Face inference, resumable JSONL execution, Pandas aggregation, tests, and
  CI-ready project structure.
- Validated the complete metric and reporting pipeline over 9,216 synthetic model-prompt evaluations.

## Claim to use only after the production run

“Evaluated six open-source LLMs across 1,500+ prompts” becomes supported when
`configs/open_source_6.json` has completed and `findings/open_source_6/run_manifest.json` shows six model
IDs, 1,536 unique prompts, 9,216 result rows, and `result_kind: measured_model_evaluation`.

Keep the manifest and reports in the repository when using measured numbers in a résumé, article, or
interview. This makes the claims reproducible and avoids presenting deterministic validation data as
real-model findings.

