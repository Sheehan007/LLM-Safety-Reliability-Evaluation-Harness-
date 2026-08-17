# Running the six-model benchmark

The production configuration is `configs/open_source_6.json`. It includes six Hugging Face chat models
at a range of parameter sizes. Some models require accepting their license on Hugging Face and setting
`HF_TOKEN`.

## Recommended environment

- Python 3.10–3.12
- A CUDA GPU with enough memory for the selected model, or Apple Silicon for the smaller entries
- Sufficient disk space for model weights and the Hugging Face cache

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml]"
python -m llm_safety_harness build-data --force
python -m llm_safety_harness run --config configs/open_source_6.json
```

Runs are resumable. If execution stops after a completed batch, rerunning the same command skips existing
`(model_id, case_id)` rows. Do not change decoding parameters midway through the same output directory;
use a new run name and output directory instead.

For constrained hardware, copy the configuration, keep one model, lower `batch_size` to 1, and test the
500-case file first by changing `dataset_path` to `data/benchmark/perturbations_500.jsonl`.

