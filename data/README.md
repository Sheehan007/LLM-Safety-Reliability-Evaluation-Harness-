# Evaluation data

The repository commits both the small curated seed set and the deterministic expanded suite.

- `seeds/seed_prompts.jsonl`: 32 human-readable seed cases across six evaluation categories.
- `benchmark/prompts.jsonl`: 1,536 generated cases (32 seeds × 48 controlled versions).
- `benchmark/perturbations_500.jsonl`: fixed 500-case stress-test slice used for robustness analysis.

Every expanded record keeps its `parent_id`, perturbation family, variation index, expected refusal
label, attack type, reference answer, and formatting constraints. Regenerate the files with:

```bash
python -m llm_safety_harness build-data --force
```

The prompts are curated test fixtures, not a claim of comprehensive safety coverage. Harmful requests
are included only to test refusal behavior; mock failures never emit actionable harmful content.

