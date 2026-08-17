# Contributing

1. Create a focused branch and keep generated model weights out of Git.
2. Add or update tests for metric, perturbation, backend, or reporting changes.
3. Run `make data`, `make test`, and `make lint` before opening a pull request.
4. Label every result set as `synthetic_validation` or `measured_model_evaluation`.
5. Never commit secrets, access tokens, unredacted personal data, or actionable harmful model outputs.

New benchmark prompts should include a stable ID, category, expected behavior, source metadata, and only
the minimum harmful detail needed to test a refusal boundary. Preserve seed-to-variation traceability.

