# portfolio-validation-v1 — Findings

**Result type:** Synthetic validation run  
**Models:** 6  
**Unique prompts:** 1,536  
**Model-prompt evaluations:** 9,216

> **Interpretation warning:** These results come from deterministic behavioral simulators. They validate the evaluation pipeline and are not measurements of the six Hugging Face models.

## Executive summary

- `mock/guarded` ranked highest on the composite safety ordering, with attack-success rate 6.8%.
- `mock/permissive` showed the most behavioral drift under controlled variations (36.9%).
- `mock/permissive` produced the largest number of flagged cases (810).
- Treat the metrics as complementary: refusal strength can improve safety while also increasing false refusals, and lexical similarity is not a semantic judge.

## Model comparison

| model_id | prompt_count | task_success | attack_success_rate | refusal_consistency | behavioral_drift | failure_count |
| --- | --- | --- | --- | --- | --- | --- |
| mock/guarded | 1536 | 0.876 | 0.068 | 0.931 | 0.183 | 307 |
| mock/balanced | 1536 | 0.870 | 0.141 | 0.856 | 0.194 | 318 |
| mock/helpful | 1536 | 0.807 | 0.268 | 0.720 | 0.284 | 487 |
| mock/brittle | 1536 | 0.678 | 0.350 | 0.660 | 0.358 | 619 |
| mock/literal | 1536 | 0.735 | 0.376 | 0.676 | 0.337 | 583 |
| mock/permissive | 1536 | 0.659 | 0.537 | 0.593 | 0.369 | 810 |

## Category breakdown

| model_id | category | prompt_count | accuracy | attack_success_rate | task_success |
| --- | --- | --- | --- | --- | --- |
| mock/balanced | factuality | 288 | 0.861 | — | 0.861 |
| mock/balanced | instruction_following | 288 | 0.882 | — | 0.882 |
| mock/balanced | jailbreak | 240 | — | 0.171 | 0.829 |
| mock/balanced | prompt_injection | 240 | — | 0.138 | 0.863 |
| mock/balanced | refusal_behavior | 192 | — | 0.109 | 0.891 |
| mock/balanced | robustness | 288 | 0.896 | — | 0.896 |
| mock/brittle | factuality | 288 | 0.674 | — | 0.674 |
| mock/brittle | instruction_following | 288 | 0.736 | — | 0.736 |
| mock/brittle | jailbreak | 240 | — | 0.358 | 0.642 |
| mock/brittle | prompt_injection | 240 | — | 0.367 | 0.633 |
| mock/brittle | refusal_behavior | 192 | — | 0.318 | 0.682 |
| mock/brittle | robustness | 288 | 0.691 | — | 0.691 |
| mock/guarded | factuality | 288 | 0.788 | — | 0.788 |
| mock/guarded | instruction_following | 288 | 0.878 | — | 0.878 |
| mock/guarded | jailbreak | 240 | — | 0.071 | 0.929 |
| mock/guarded | prompt_injection | 240 | — | 0.067 | 0.933 |
| mock/guarded | refusal_behavior | 192 | — | 0.068 | 0.932 |
| mock/guarded | robustness | 288 | 0.830 | — | 0.830 |
| mock/helpful | factuality | 288 | 0.858 | — | 0.858 |
| mock/helpful | instruction_following | 288 | 0.858 | — | 0.858 |
| mock/helpful | jailbreak | 240 | — | 0.250 | 0.750 |
| mock/helpful | prompt_injection | 240 | — | 0.246 | 0.754 |
| mock/helpful | refusal_behavior | 192 | — | 0.318 | 0.682 |
| mock/helpful | robustness | 288 | 0.882 | — | 0.882 |
| mock/literal | factuality | 288 | 0.823 | — | 0.823 |
| mock/literal | instruction_following | 288 | 0.809 | — | 0.809 |
| mock/literal | jailbreak | 240 | — | 0.283 | 0.717 |
| mock/literal | prompt_injection | 240 | — | 0.471 | 0.529 |
| mock/literal | refusal_behavior | 192 | — | 0.375 | 0.625 |
| mock/literal | robustness | 288 | 0.833 | — | 0.833 |
| mock/permissive | factuality | 288 | 0.837 | — | 0.837 |
| mock/permissive | instruction_following | 288 | 0.806 | — | 0.806 |
| mock/permissive | jailbreak | 240 | — | 0.500 | 0.500 |
| mock/permissive | prompt_injection | 240 | — | 0.554 | 0.446 |
| mock/permissive | refusal_behavior | 192 | — | 0.562 | 0.438 |
| mock/permissive | robustness | 288 | 0.792 | — | 0.792 |

## Fixed 500-perturbation stress test

| model_id | prompt_count | task_success | attack_success_rate | consistency_under_variation | behavioral_drift | failure_count |
| --- | --- | --- | --- | --- | --- | --- |
| mock/guarded | 500 | 0.884 | 0.068 | 0.827 | 0.173 | 96 |
| mock/balanced | 500 | 0.862 | 0.145 | 0.787 | 0.213 | 112 |
| mock/helpful | 500 | 0.818 | 0.241 | 0.733 | 0.267 | 146 |
| mock/brittle | 500 | 0.680 | 0.336 | 0.631 | 0.369 | 202 |
| mock/literal | 500 | 0.702 | 0.427 | 0.641 | 0.359 | 207 |
| mock/permissive | 500 | 0.642 | 0.586 | 0.630 | 0.370 | 264 |

## Failure taxonomy

| model_id | failure_reason | count |
| --- | --- | --- |
| mock/balanced | high_behavioral_drift | 301 |
| mock/balanced | incorrect | 104 |
| mock/balanced | attack_success | 95 |
| mock/balanced | missed_refusal | 62 |
| mock/balanced | instruction_violation | 34 |
| mock/balanced | false_refusal | 9 |
| mock/brittle | high_behavioral_drift | 552 |
| mock/brittle | incorrect | 259 |
| mock/brittle | attack_success | 235 |
| mock/brittle | missed_refusal | 147 |
| mock/brittle | instruction_violation | 76 |
| mock/guarded | high_behavioral_drift | 283 |
| mock/guarded | incorrect | 145 |
| mock/guarded | attack_success | 46 |
| mock/guarded | instruction_violation | 35 |
| mock/guarded | missed_refusal | 30 |
| mock/guarded | false_refusal | 23 |
| mock/helpful | high_behavioral_drift | 438 |
| mock/helpful | attack_success | 180 |
| mock/helpful | missed_refusal | 121 |
| mock/helpful | incorrect | 116 |
| mock/helpful | instruction_violation | 41 |
| mock/literal | high_behavioral_drift | 519 |
| mock/literal | attack_success | 253 |
| mock/literal | incorrect | 154 |
| mock/literal | missed_refusal | 140 |
| mock/literal | instruction_violation | 55 |
| mock/permissive | high_behavioral_drift | 568 |
| mock/permissive | attack_success | 361 |
| mock/permissive | missed_refusal | 228 |
| mock/permissive | incorrect | 163 |
| mock/permissive | instruction_violation | 56 |

## Reading these results

Attack-success rate is lower-is-better. Task success, refusal consistency, and accuracy are higher-is-better. Behavioral drift is the mean lexical cosine distance from each seed prompt's baseline response, so lower is better. See [`docs/methodology.md`](../../docs/methodology.md) for definitions, caveats, and the threat model.
