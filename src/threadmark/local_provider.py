"""Keyless, local-only Hugging Face causal-language-model provider.

Optional torch/transformers dependencies and local model files are loaded only
when needed. This adapter never downloads weights or runs remote model code.
It records token usage; local hardware and electricity costs remain unpriced.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version as package_version
import json
import logging
from math import isfinite
from pathlib import Path
import re
import time
from typing import Any

from .providers import Completion, ProviderError
from .schema import canonical, parse_json


_CONSTRAINT_VERSION = "0.11.3"


class _ConstraintLogRedactor(logging.Filter):
    """Suppress raw prefix/traceback content in LMFE's own diagnostics only."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def filter(self, record: logging.LogRecord) -> bool:
        source = record.pathname.replace("\\", "/")
        if (
            record.name == "lmformatenforcer"
            or record.name.startswith("lmformatenforcer.")
            or "/lmformatenforcer/" in source
        ):
            self.failed |= record.levelno >= logging.ERROR
            record.msg = (
                "Workflow constraint diagnostic redacted to protect input text."
            )
            record.args = ()
            record.exc_info = record.exc_text = record.stack_info = None
        return True


@contextmanager
def _constraint_diagnostics():
    # The pinned LMFE version calls logging.exception through the root logger.
    # Filter only LMFE-origin records; never disable unrelated application logs.
    redactor = _ConstraintLogRedactor()
    loggers = [
        logging.getLogger(name)
        for name in ("", "lmformatenforcer", "lmformatenforcer.tokenenforcer")
    ]
    for logger in loggers:
        logger.addFilter(redactor)
    try:
        yield redactor
    finally:
        for logger in loggers:
            logger.removeFilter(redactor)


def _file_sha256(path: Path) -> str:
    checksum = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _local_json(path: Path) -> tuple[dict, str]:
    with path.open("rb") as handle:
        raw = handle.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("Local configuration exceeds its size limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Local configuration must be a JSON object")
    return value, sha256(raw).hexdigest()


class LocalTransformersProvider:
    """Run a frozen local model with its native chat template and greedy decode.

    ``model_path`` must contain a previously downloaded tokenizer, config and
    safetensors weights. Downloads must be performed separately by the caller.
    One instance supports sequential calls. CPU uses float32; CUDA uses bfloat16
    when the selected device supports it, otherwise float32. ``max_context_tokens``
    can lower the model's native context limit but cannot extend it.

    ``adapter_path`` optionally supplies an ordinary linear LoRA CAUSAL_LM
    adapter. Its recorded base path is never followed. Only the caller's local
    base and tokenizer are loaded; adapter tensors must exactly cover matching
    target modules. This proves structural compatibility, not that these are
    the original training-base weights. Pin that provenance separately. Other
    PEFT types, extra saved modules and modified layer layouts are unsupported.

    ``structured_workflow`` opts into request-specific JSON Schema constraints
    for WorkflowMemory updater calls only. It needs the ``local-structured``
    extra. ``max_workflow_value_chars`` must match the memory's Settings.
    Grammar/schema validation does not verify source truth or replace reducer
    validation. Constraint failures never fall back to unconstrained decoding.
    Sequential callers may change the Boolean flag between calls; tokenizer
    preprocessing is cached, but schemas and parser state are never reused.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device: str = "cuda",
        max_context_tokens: int | None = None,
        adapter_path: str | Path | None = None,
        structured_workflow: bool = False,
        max_workflow_value_chars: int = 256,
    ) -> None:
        if not isinstance(model_path, (str, Path)) or not str(model_path).strip():
            raise ValueError("model_path must identify a local model directory")
        if adapter_path is not None and (
            not isinstance(adapter_path, (str, Path)) or not str(adapter_path).strip()
        ):
            raise ValueError("adapter_path must identify a local adapter directory")
        if type(structured_workflow) is not bool:
            raise ValueError("structured_workflow must be a Boolean")
        if (
            type(max_workflow_value_chars) is not int
            or not 1 <= max_workflow_value_chars <= 1_000_000
        ):
            raise ValueError(
                "max_workflow_value_chars must be an integer from 1 to 1000000"
            )
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
        self.adapter_path = (
            Path(adapter_path).expanduser().resolve()
            if adapter_path is not None
            else None
        )
        self.model = model
        self.structured_workflow = structured_workflow
        self.max_workflow_value_chars = max_workflow_value_chars
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
        self._adapter_identity: dict[str, Any] | None = None
        self._constraint_tokenizer_data: Any = None

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
            "adapter": deepcopy(self._adapter_identity),
            "structured_workflow": self.structured_workflow,
            "max_workflow_value_chars": self.max_workflow_value_chars,
        }

    def _workflow_prefix(self, schema: dict, eos_tokens: list[int]):
        """Build a fresh grammar; cache only immutable tokenizer preprocessing.

        Integration follows the project's primary Transformers API:
        https://github.com/noamgat/lm-format-enforcer
        """
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.characterlevelparser import (
                CharacterLevelParserConfig,
            )
            from lmformatenforcer.integrations.transformers import (
                build_token_enforcer_tokenizer_data,
                build_transformers_prefix_allowed_tokens_fn,
            )

            if package_version("lm-format-enforcer") != _CONSTRAINT_VERSION:
                raise ValueError("Unsupported constraint engine version")
            with _constraint_diagnostics() as diagnostics:
                if self._constraint_tokenizer_data is None:
                    self._constraint_tokenizer_data = (
                        build_token_enforcer_tokenizer_data(self._tokenizer)
                    )
                tokenizer_data = self._constraint_tokenizer_data
                # The model may declare additional EOS tokens beyond the
                # tokenizer's primary EOS. Keep grammar and generation aligned.
                tokenizer_data.eos_token_id = list(eos_tokens)
                config = CharacterLevelParserConfig(
                    alphabet=tokenizer_data.tokenizer_alphabet,
                    max_consecutive_whitespaces=12,
                    force_json_field_order=False,
                    max_json_array_length=128,
                )
                parser = JsonSchemaParser(schema, config=config)
                prefix = build_transformers_prefix_allowed_tokens_fn(
                    tokenizer_data, parser
                )
                # LMFE 0.11.3 replaces this config in TokenEnforcer.__init__.
                # Restore our explicit settings instead of ambient LMFE_* env.
                parser.config = config
            if diagnostics.failed:
                raise ValueError("Constraint engine initialization failed")
        except Exception as exc:
            raise ProviderError(
                f"Workflow constraint initialization failed ({type(exc).__name__}); "
                "install the local-structured extra. No fallback or retry was made",
                fatal=False,
            ) from None

        def constrained(batch_id, token_ids):
            try:
                with _constraint_diagnostics() as diagnostics:
                    allowed = prefix(batch_id, token_ids)
                if diagnostics.failed:
                    raise ValueError("Constraint engine reported an error")
                return allowed
            except Exception as exc:
                raise ProviderError(
                    f"Workflow constraint engine failed ({type(exc).__name__}); "
                    "no fallback or retry was made",
                    request_started=True,
                    fatal=False,
                ) from None

        return constrained

    def _adapter_specification(self) -> dict[str, Any] | None:
        if self.adapter_path is None:
            return None
        directory = self.adapter_path
        if not directory.is_dir():
            raise ValueError("Local adapter directory is unavailable")
        config_path = directory / "adapter_config.json"
        weights_path = directory / "adapter_model.safetensors"
        for path in (config_path, weights_path):
            if not path.is_file() or not path.resolve().is_relative_to(directory):
                raise ValueError("Adapter requires local configuration and safetensors")
        config, config_hash = _local_json(config_path)
        base, base_hash = _local_json(self.model_path / "config.json")
        known_options = {
            "peft_type",
            "task_type",
            "model_type",
            "inference_mode",
            "base_model_name_or_path",
            "revision",
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "bias",
            "init_lora_weights",
            "use_rslora",
            "megatron_core",
            "qalora_group_size",
            "auto_mapping",
            "modules_to_save",
            "target_parameters",
            "exclude_modules",
            "layers_to_transform",
            "layers_pattern",
            "rank_pattern",
            "alpha_pattern",
            "layer_replication",
            "trainable_token_indices",
            "loftq_config",
            "megatron_config",
            "corda_config",
            "eva_config",
            "use_dora",
            "use_qalora",
            "lora_bias",
            "fan_in_fan_out",
        }
        if set(config) - known_options:
            raise ValueError("Unsupported adapter configuration options")
        model_type = base.get("model_type")
        if not isinstance(model_type, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", model_type
        ):
            raise ValueError("Base configuration must declare a model type")
        if config.get("model_type", model_type) != model_type:
            raise ValueError("Adapter and base model types differ")
        if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
            raise ValueError("Only CAUSAL_LM LoRA adapters are supported")
        # These variants need different state-key or shape rules. Do not allow
        # PEFT to silently initialize unmatched tensors or load external assets.
        for key in (
            "auto_mapping",
            "modules_to_save",
            "target_parameters",
            "exclude_modules",
            "layers_to_transform",
            "layers_pattern",
            "rank_pattern",
            "alpha_pattern",
            "layer_replication",
            "trainable_token_indices",
            "loftq_config",
            "megatron_config",
            "corda_config",
            "eva_config",
            "use_dora",
            "use_qalora",
            "lora_bias",
            "fan_in_fan_out",
        ):
            if config.get(key) not in (None, False, {}, []):
                raise ValueError("Unsupported LoRA adapter variant")
        if config.get("bias", "none") != "none":
            raise ValueError("Adapters with additional bias weights are unsupported")
        if type(config.get("init_lora_weights", True)) is not bool:
            raise ValueError("Adapters requiring special initializers are unsupported")
        rank = config.get("r")
        alpha, dropout = config.get("lora_alpha"), config.get("lora_dropout", 0.0)
        if type(rank) is not int or not 1 <= rank <= 4096:
            raise ValueError("Invalid LoRA rank")
        if (
            type(alpha) not in (int, float)
            or not isfinite(alpha)
            or not 0 < alpha <= 1_000_000
            or type(dropout) not in (int, float)
            or not isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("Invalid LoRA scaling or dropout")
        targets = config.get("target_modules")
        if (
            not isinstance(targets, list)
            or not 1 <= len(targets) <= 1024
            or any(
                not isinstance(t, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,255}", t)
                for t in targets
            )
            or len(set(targets)) != len(targets)
            or type(config.get("use_rslora", False)) is not bool
        ):
            raise ValueError("Adapter requires explicit unique target module names")
        weights_hash = _file_sha256(weights_path)
        identity = {
            "adapter_id": "sha256:"
            + sha256(
                canonical(
                    {
                        "config": config_hash,
                        "weights": weights_hash,
                    }
                ).encode("utf-8")
            ).hexdigest(),
            "config_sha256": config_hash,
            "weights_sha256": weights_hash,
            "base_config_sha256": base_hash,
            "base_model_type": model_type,
            "compatibility": (
                "exact_lora_target_shapes; training_base_revision_unverified"
            ),
        }
        return {
            "identity": identity,
            "config_path": config_path,
            "weights_path": weights_path,
            # Only known scalar options enter PEFT. In particular, never pass
            # through base_model_name_or_path, revision or external initializers.
            "config": {
                "task_type": "CAUSAL_LM",
                "inference_mode": True,
                "r": rank,
                "lora_alpha": alpha,
                "lora_dropout": dropout,
                "target_modules": targets,
                "bias": "none",
                "use_rslora": config.get("use_rslora", False),
                "base_model_name_or_path": str(self.model_path),
            },
        }

    def _attach_adapter(self, model: Any, specification: dict[str, Any]) -> Any:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file

        identity, config = specification["identity"], specification["config"]
        if getattr(model.config, "model_type", None) != identity["base_model_type"]:
            raise ValueError(
                "Loaded base model type differs from its local configuration"
            )
        targets, rank = config["target_modules"], config["r"]
        expected, matched = {}, set()
        for name, module in model.named_modules():
            selectors = {
                target
                for target in targets
                if name == target or name.endswith("." + target)
            }
            if not selectors:
                continue
            shape = getattr(getattr(module, "weight", None), "shape", ())
            if len(shape) != 2 or tuple(shape) != (
                getattr(module, "out_features", None),
                getattr(module, "in_features", None),
            ):
                raise ValueError("Adapter target must be a linear weight matrix")
            matched.update(selectors)
            prefix = "base_model.model." + name
            expected[prefix + ".lora_A.weight"] = (rank, shape[1])
            expected[prefix + ".lora_B.weight"] = (shape[0], rank)
        if matched != set(targets) or not expected:
            raise ValueError("Adapter targets do not exist in the supplied base")
        # Use no PEFT/Hugging Face file resolver. Some PEFT versions can probe
        # the Hub for missing weights even with local_files_only=True. Reading
        # the exact safetensors file and attaching in-memory has no such fallback.
        weights = load_file(str(specification["weights_path"]), device="cpu")
        if set(weights) != set(expected) or any(
            tuple(weights[name].shape) != shape or not weights[name].is_floating_point()
            for name, shape in expected.items()
        ):
            raise ValueError("Adapter tensors do not exactly match the supplied base")
        adapted = get_peft_model(model, LoraConfig(**config), adapter_name="default")
        result = set_peft_model_state_dict(
            adapted,
            weights,
            adapter_name="default",
            ignore_mismatched_sizes=False,
        )
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError("PEFT did not load every adapter tensor")
        if (
            _file_sha256(specification["weights_path"]) != identity["weights_sha256"]
            or _file_sha256(specification["config_path"]) != identity["config_sha256"]
        ):
            raise ValueError("Local adapter files changed during loading")
        return adapted

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        if not self.model_path.is_dir():
            raise ProviderError(
                "Local model directory is unavailable; no download or retry was made"
            )
        if (self.model_path / "adapter_config.json").exists():
            raise ProviderError(
                "model_path must identify the base model; supply its adapter separately"
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
            adapter = self._adapter_specification()
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
            if adapter is not None:
                model = self._attach_adapter(model, adapter)
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
        self._adapter_identity = deepcopy(adapter["identity"]) if adapter else None
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
        structured = self.structured_workflow
        if type(structured) is not bool:
            raise ValueError("structured_workflow must be a Boolean")
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
            "adapter_requested": self.adapter_path is not None,
            "adapter": deepcopy(self._adapter_identity),
            "structured_workflow": structured,
            "workflow_schema_sha256": None,
            "constraint_engine": None,
            "constraint_setup_seconds": None,
        }
        try:
            schema = validator = None
            if structured:
                try:
                    from jsonschema import Draft202012Validator
                    from .workflow_schema import workflow_update_schema

                    schema = workflow_update_schema(
                        content, max_value_chars=self.max_workflow_value_chars
                    )
                    Draft202012Validator.check_schema(schema)
                    validator = Draft202012Validator(schema)
                except Exception as exc:
                    raise ProviderError(
                        "Workflow constraint request validation failed "
                        f"({type(exc).__name__}); expected an updater request and "
                        "the local-structured extra. No fallback or retry was made",
                        fatal=False,
                    ) from None
                record["workflow_schema_sha256"] = sha256(
                    canonical(schema).encode("utf-8")
                ).hexdigest()
                record["constraint_engine"] = (
                    "lm-format-enforcer/" + _CONSTRAINT_VERSION
                )
            self._ensure_model()
            record["adapter"] = deepcopy(self._adapter_identity)
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
            generation_options = {}
            if structured:
                constraint_started = time.perf_counter()
                generation_options["prefix_allowed_tokens_fn"] = self._workflow_prefix(
                    schema, eos_tokens
                )
                record["constraint_setup_seconds"] = (
                    time.perf_counter() - constraint_started
                )
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
                    **generation_options,
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
            if structured:
                try:
                    # LMFE errors can terminate with EOS. Require strict JSON
                    # (including duplicate-key rejection) and independently
                    # validate its schema before exposing a successful result.
                    validator.validate(parse_json(text))
                except Exception as exc:
                    record["status"] = "invalid_structured_output"
                    raise ProviderError(
                        "Workflow constrained output failed strict validation "
                        f"({type(exc).__name__}); no repair, fallback or retry was made",
                        request_started=True,
                        fatal=False,
                    ) from None
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
