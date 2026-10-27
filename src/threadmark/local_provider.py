"""Keyless, local-only Hugging Face causal-language-model provider.

Optional torch/transformers dependencies and local model files are loaded only
when needed. This adapter never downloads weights or runs remote model code.
It records token usage; local hardware and electricity costs remain unpriced.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import time
from typing import Any

from .providers import Completion, ProviderError
from .schema import canonical


class LocalTransformersProvider:
    """Run a frozen local model with its native chat template and greedy decode.

    ``model_path`` must contain a previously downloaded tokenizer, config and
    safetensors weights. Downloads must be performed separately by the caller.
    One instance supports sequential calls. CPU uses float32; CUDA uses bfloat16
    when the selected device supports it, otherwise float32. ``max_context_tokens``
    can lower the model's native context limit but cannot extend it.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device: str = "cuda",
        max_context_tokens: int | None = None,
    ) -> None:
        if not isinstance(model_path, (str, Path)) or not str(model_path).strip():
            raise ValueError("model_path must identify a local model directory")
        if not isinstance(model, str) or not re.fullmatch(
            r"[A-Za-z0-9_.:/-]{1,160}", model
        ):
            raise ValueError("Invalid local model identifier")
        if not isinstance(device, str) or not re.fullmatch(
            r"cpu|cuda(?::[0-9]+)?", device
        ):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        if max_context_tokens is not None and (
            type(max_context_tokens) is not int
            or not 1 <= max_context_tokens <= 1_000_000
        ):
            raise ValueError(
                "max_context_tokens must be an integer between 1 and 1000000"
            )
        self.model_path = Path(model_path).expanduser().resolve()
        self.model = model
        self.reasoning_effort = "none"
        self.device = device
        self.max_context_tokens = max_context_tokens
        self.native_context_tokens: int | None = None
        self.context_limit: int | None = None
        self.dtype: str | None = None
        self.model_load_seconds = 0.0
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._calls: list[dict[str, Any]] = []

    @property
    def calls(self) -> list[dict[str, Any]]:
        return deepcopy(self._calls)

    def load(self) -> dict[str, Any]:
        """Load the local model once and return shared initialization metadata.

        This makes cold loading explicit before comparing context strategies.
        It performs no generation and does not add an inference call record.
        """
        already_loaded = self._model is not None
        self._ensure_model()
        return {
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
            "native_context_tokens": self.native_context_tokens,
            "context_limit": self.context_limit,
            "model_load_seconds": self.model_load_seconds,
            "already_loaded": already_loaded,
            "api_charge_usd": 0.0,
            "hardware_cost_usd": None,
            "cost_scope": "local_compute_unpriced",
        }

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        if not self.model_path.is_dir():
            raise ProviderError(
                "Local model directory is unavailable; no download or retry was made"
            )
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=False,
                token=False,
            )
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError("A native chat template is required")
        except Exception as exc:
            raise ProviderError(
                f"Local tokenizer loading failed ({type(exc).__name__}); no download or retry was made"
            ) from None
        self._tokenizer = tokenizer

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter()
        self._ensure_tokenizer()
        try:
            import torch
            from transformers import AutoModelForCausalLM

            if self.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise ValueError(
                        "CUDA is unavailable; choose device='cpu' explicitly"
                    )
                with torch.cuda.device(self.device):
                    use_bfloat16 = torch.cuda.is_bf16_supported()
                dtype = torch.bfloat16 if use_bfloat16 else torch.float32
                dtype_name = "bfloat16" if use_bfloat16 else "float32"
            else:
                dtype, dtype_name = torch.float32, "float32"
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                token=False,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
            limits = [
                getattr(model.config, name, None)
                for name in (
                    "max_position_embeddings",
                    "n_positions",
                    "max_sequence_length",
                    "seq_length",
                )
            ]
            limits = [
                value
                for value in limits
                if type(value) is int and 1 <= value <= 1_000_000
            ]
            if not limits:
                raise ValueError(
                    "Model config does not declare a supported native context limit"
                )
            native_context = min(limits)
            model.to(self.device)
            model.eval()
            model.requires_grad_(False)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
        except Exception as exc:
            raise ProviderError(
                f"Local model loading failed ({type(exc).__name__}); no download or retry was made"
            ) from None
        self._torch, self._model = torch, model
        self.native_context_tokens = native_context
        self.context_limit = min(
            native_context, self.max_context_tokens or native_context
        )
        self.dtype = dtype_name
        self.model_load_seconds = time.perf_counter() - started

    def count_tokens(self, text: str) -> int:
        """Count raw text with the local model tokenizer, without loading weights.

        This count excludes chat-template overhead. ``complete`` separately
        checks the full native chat prompt and reserved generated-token space.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._ensure_tokenizer()
        try:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        except Exception as exc:
            raise ProviderError(
                f"Local token counting failed ({type(exc).__name__}); no retry was made"
            ) from None

    def _synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize(self.device)

    def _stop_tokens(self) -> tuple[list[int], int]:
        config = getattr(self._model, "generation_config", None)
        eos = getattr(config, "eos_token_id", None)
        if eos is None:
            eos = getattr(self._tokenizer, "eos_token_id", None)
        values = eos if isinstance(eos, list) else [eos]
        if not values or any(type(value) is not int or value < 0 for value in values):
            raise ProviderError(
                "Local model requires valid native EOS token IDs; no retry was made"
            )
        pad = getattr(self._tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(config, "pad_token_id", None)
        if pad is None:
            pad = values[0]
        if type(pad) is not int or pad < 0:
            raise ProviderError(
                "Local model has an invalid padding token ID; no retry was made"
            )
        return values, pad

    def complete(
        self, *, instructions: str, content: dict, max_output_tokens: int
    ) -> Completion:
        """Generate once; context overflow or model failures never trigger retries.

        Completion cost fields are unset because local compute is unpriced.
        Call metadata separately reports zero hosted API charge, not zero total
        cost. Diagnostics contain no prompt, input data, output text or paths.
        """
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("instructions must be a nonempty string")
        if not isinstance(content, dict):
            raise TypeError("content must be a JSON object")
        if (
            type(max_output_tokens) is not int
            or not 1 <= max_output_tokens <= 1_000_000
        ):
            raise ValueError(
                "max_output_tokens must be an integer between 1 and 1000000"
            )
        serialized = canonical(content)
        started = time.perf_counter()
        record: dict[str, Any] = {
            "model": self.model,
            "device": self.device,
            "status": "failed",
            "request_started": False,
            "usage": None,
            "elapsed_seconds": None,
            "generation_seconds": None,
            "api_charge_usd": 0.0,
            "hardware_cost_usd": None,
            "cost_scope": "local_compute_unpriced",
            "estimated_cost_usd": None,
            "budget_cost_usd": None,
        }
        try:
            self._ensure_model()
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": serialized},
            ]
            template_options = {}
            if "enable_thinking" in getattr(self._tokenizer, "chat_template", ""):
                template_options["enable_thinking"] = False
            batch = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **template_options,
            )
            input_ids = batch["input_ids"]
            if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
                raise ProviderError(
                    "Native chat template returned an invalid token batch; no retry was made"
                )
            input_count = input_ids.shape[1]
            record.update(
                input_tokens=input_count,
                context_limit=self.context_limit,
                dtype=self.dtype,
            )
            if input_count + max_output_tokens > self.context_limit:
                raise ProviderError(
                    "Full native chat prompt plus output reservation exceeds the local context limit; no truncation or retry was made",
                    fatal=False,
                )
            inputs = {"input_ids": input_ids.to(self.device)}
            attention = batch.get("attention_mask")
            inputs["attention_mask"] = (
                attention if attention is not None else self._torch.ones_like(input_ids)
            ).to(self.device)
            eos_tokens, pad_token = self._stop_tokens()
            self._synchronize()
            generation_started = time.perf_counter()
            record["request_started"] = True
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=max_output_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    return_dict_in_generate=False,
                    eos_token_id=eos_tokens,
                    pad_token_id=pad_token,
                )
            self._synchronize()
            record["generation_seconds"] = time.perf_counter() - generation_started
            if (
                generated.ndim != 2
                or generated.shape[0] != 1
                or generated.shape[1] < input_count
            ):
                raise ProviderError(
                    "Local generation returned an invalid token batch; no retry was made",
                    request_started=True,
                )
            new_ids = generated[0, input_count:].tolist()
            if len(new_ids) > max_output_tokens:
                raise ProviderError(
                    "Local generation exceeded its output reservation; no retry was made",
                    request_started=True,
                )
            usage = {
                "input_tokens": input_count,
                "output_tokens": len(new_ids),
                "input_tokens_details": {"cached_tokens": 0},
            }
            record["usage"] = usage
            if not new_ids or new_ids[-1] not in eos_tokens:
                record["status"] = "incomplete"
                raise ProviderError(
                    "Local generation ended without a native EOS token; output may be truncated and no retry was made",
                    request_started=True,
                    fatal=False,
                )
            text = self._tokenizer.decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            if not text:
                raise ProviderError(
                    "Local generation returned no readable text; no retry was made",
                    request_started=True,
                    fatal=False,
                )
            record["status"] = "completed"
            return Completion(
                text=text,
                usage=deepcopy(usage),
                elapsed_seconds=time.perf_counter() - started,
                model=self.model,
            )
        except ProviderError as exc:
            record["error"] = str(exc)
            raise
        except Exception as exc:
            failure = ProviderError(
                f"Local generation failed ({type(exc).__name__}); no retry was made",
                request_started=record["request_started"],
            )
            record["error"] = str(failure)
            raise failure from None
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            self._calls.append(record)
