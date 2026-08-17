from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..schemas import EvaluationCase, ModelConfig
from .base import ModelBackend


class HuggingFaceBackend(ModelBackend):
    """Batched local inference through Transformers and PyTorch."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face backend requires the ML extras. "
                "Install them with: pip install -e '.[ml]'"
            ) from exc

        self.torch = torch
        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
        }
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if config.revision:
            tokenizer_kwargs["revision"] = config.revision
            model_kwargs["revision"] = config.revision

        self.tokenizer = AutoTokenizer.from_pretrained(config.id, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = self._resolve_dtype(config.dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if config.device == "auto":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(config.id, **model_kwargs)
        if config.device != "auto":
            self.model.to(config.device)
        self.model.eval()

    def _resolve_dtype(self, name: str):
        if name == "auto":
            if self.torch.cuda.is_available():
                return self.torch.float16
            if getattr(self.torch.backends, "mps", None) and self.torch.backends.mps.is_available():
                return self.torch.float16
            return self.torch.float32
        mapping = {
            "float32": self.torch.float32,
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported dtype: {name}")
        return mapping[name]

    def _format_prompt(self, case: EvaluationCase) -> str:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": case.prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (ValueError, TypeError):
                pass
        return f"System: {self.config.system_prompt}\nUser: {case.prompt}\nAssistant:"

    def generate(self, cases: Sequence[EvaluationCase]) -> list[str]:
        prompts = [self._format_prompt(case) for case in cases]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(getattr(self.tokenizer, "model_max_length", 4096), 4096),
        )
        model_device = next(self.model.parameters()).device
        encoded = {name: tensor.to(model_device) for name, tensor in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        do_sample = self.config.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.config.temperature

        with self.torch.inference_mode():
            generated = self.model.generate(**encoded, **generation_kwargs)
        suffixes = generated[:, input_width:]
        return self.tokenizer.batch_decode(suffixes, skip_special_tokens=True)

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

