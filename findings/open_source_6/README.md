# Six-model findings — pending execution

No empirical model results are committed yet. Run:

```bash
python -m llm_safety_harness run --config configs/open_source_6.json
```

When complete, this directory will contain raw JSONL, scored CSVs, aggregate tables, a manifest, a main
findings report, and one failure report per model. The separation prevents the deterministic validation
run from being misrepresented as real open-source-model evidence.

