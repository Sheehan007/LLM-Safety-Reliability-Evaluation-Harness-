# Limitations and responsible use

- The included findings are a deterministic pipeline validation, not empirical measurements of the six
  open-source models.
- The prompt set is English-only, single-turn, text-only, and deliberately small at the seed level.
- Surface perturbations test stability but are not a substitute for expert red teaming.
- The metric layer uses transparent heuristics. It should be paired with human annotations before making
  deployment decisions.
- Model behavior depends on weights, revisions, inference libraries, chat templates, quantization, and
  hardware. Pin revisions for publication-quality comparisons.
- Safety and helpfulness are multi-objective. High refusal rates can hide over-refusal on benign inputs.
- No benchmark result should be treated as a safety guarantee or used as the sole release gate.

The unsafe prompts are present for defensive evaluation. Do not extend the fixture responses with
actionable harmful content; use placeholders or redacted outputs in committed artifacts.

