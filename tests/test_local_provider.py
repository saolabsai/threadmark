"""Mocked local-provider tests require neither torch nor model downloads."""

from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import pytest

from threadmark.local_provider import LocalTransformersProvider
from threadmark.providers import ProviderError


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values[:]


class FakeTensor:
    ndim = 2

    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self

    def __getitem__(self, item):
        row, selection = item
        return FakeVector(self.values[row][selection])


class FakeTokenizer:
    chat_template = "native-model-template"
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.templates = []
        self.decoded = []
        self.output_text = '{"facts":[]}'
        self.input_length = 3

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(range(len(text)))

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return {
            "input_ids": FakeTensor([[11] * self.input_length]),
            "attention_mask": FakeTensor([[1] * self.input_length]),
        }

    def decode(self, ids, **kwargs):
        self.decoded.append((ids, kwargs))
        return self.output_text


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=64)
        self.generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=0)
        self.device = None
        self.evaluating = False
        self.frozen = False
        self.generations = []
        self.new_ids = [21, 22, 2]
        self.failure = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self

    def requires_grad_(self, enabled):
        self.frozen = not enabled
        return self

    def generate(self, **kwargs):
        self.generations.append(kwargs)
        if self.failure is not None:
            raise self.failure
        original = kwargs["input_ids"].values[0]
        return FakeTensor([original + self.new_ids])


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    tokenizer, model = FakeTokenizer(), FakeModel()
    tokenizer_loads, model_loads, synchronizations = [], [], []

    def load_tokenizer(*args, **kwargs):
        tokenizer_loads.append((args, kwargs))
        return tokenizer

    def load_model(*args, **kwargs):
        model_loads.append((args, kwargs))
        return model

    cuda = SimpleNamespace(
        available=True,
        bfloat=True,
        is_available=lambda: cuda.available,
        is_bf16_supported=lambda: cuda.bfloat,
        device=lambda device: nullcontext(),
        synchronize=lambda device: synchronizations.append(device),
    )
    torch = SimpleNamespace(
        cuda=cuda,
        float32="float32",
        bfloat16="bfloat16",
        inference_mode=lambda: nullcontext(),
        ones_like=lambda tensor: FakeTensor([[1] * tensor.shape[1]]),
    )
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return SimpleNamespace(
        path=tmp_path,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        tokenizer_loads=tokenizer_loads,
        model_loads=model_loads,
        synchronizations=synchronizations,
    )


def complete(provider, **kwargs):
    return provider.complete(
        instructions="Use only supplied facts.",
        content={"z": 1, "a": "PUBLIC_WORKFLOW_TEXT"},
        max_output_tokens=kwargs.pop("max_output_tokens", 8),
        **kwargs,
    )


def test_constructor_is_lightweight_and_imports_no_optional_dependencies(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    provider = LocalTransformersProvider(tmp_path)
    assert provider.calls == []
    assert provider.context_limit is None
    assert provider.reasoning_effort == "none"


def test_native_counter_loads_only_local_tokenizer_and_no_weights(runtime):
    provider = LocalTransformersProvider(runtime.path)
    assert provider.count_tokens("abc") == 3
    assert provider.count_tokens("de") == 2
    assert len(runtime.tokenizer_loads) == 1
    assert runtime.model_loads == []
    assert runtime.tokenizer_loads[0][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "token": False,
    }


def test_explicit_load_separates_shared_initialization_from_inference(runtime):
    provider = LocalTransformersProvider(runtime.path, device="cpu")
    metadata = provider.load()
    assert metadata["already_loaded"] is False
    assert metadata["model_load_seconds"] >= 0
    assert metadata["context_limit"] == metadata["native_context_tokens"] == 64
    assert metadata["dtype"] == "float32"
    assert metadata["hardware_cost_usd"] is None
    assert provider.calls == []
    repeated = provider.load()
    assert repeated["already_loaded"] is True
    assert repeated["model_load_seconds"] == metadata["model_load_seconds"]
    complete(provider)
    assert len(runtime.model_loads) == 1
    assert len(runtime.tokenizer_loads) == 1
    assert len(provider.calls) == 1


def test_cpu_native_chat_greedy_decode_frozen_model_and_actual_usage(runtime):
    provider = LocalTransformersProvider(runtime.path, device="cpu")
    result = complete(provider)
    assert result.text == '{"facts":[]}'
    assert result.usage == {
        "input_tokens": 3,
        "output_tokens": 3,
        "input_tokens_details": {"cached_tokens": 0},
    }
    assert result.estimated_cost_usd is None
    assert result.budget_cost_usd is None
    assert result.elapsed_seconds >= 0
    assert runtime.model.evaluating and runtime.model.frozen
    assert runtime.model.device == "cpu"
    loaded = runtime.model_loads[0][1]
    assert loaded["local_files_only"] is True
    assert loaded["trust_remote_code"] is False
    assert loaded["use_safetensors"] is True
    assert loaded["torch_dtype"] == "float32"
    assert loaded["attn_implementation"] == "sdpa"
    assert loaded["token"] is False
    messages, template_kwargs = runtime.tokenizer.templates[0]
    assert messages == [
        {"role": "system", "content": "Use only supplied facts."},
        {"role": "user", "content": '{"a":"PUBLIC_WORKFLOW_TEXT","z":1}'},
    ]
    assert template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    generation = runtime.model.generations[0]
    assert generation["do_sample"] is False
    assert generation["num_beams"] == 1
    assert generation["max_new_tokens"] == 8
    assert generation["eos_token_id"] == [2]
    assert runtime.synchronizations == []


@pytest.mark.parametrize("bfloat, expected", [(True, "bfloat16"), (False, "float32")])
def test_cuda_dtype_and_synchronized_timing(runtime, bfloat, expected):
    runtime.torch.cuda.bfloat = bfloat
    provider = LocalTransformersProvider(runtime.path, device="cuda:0")
    complete(provider)
    assert runtime.model_loads[0][1]["torch_dtype"] == expected
    # Once after model transfer, then before and after generation.
    assert runtime.synchronizations == ["cuda:0", "cuda:0", "cuda:0"]
    assert provider.calls[0]["generation_seconds"] >= 0


def test_cuda_does_not_silently_fall_back_to_cpu(runtime):
    runtime.torch.cuda.available = False
    provider = LocalTransformersProvider(runtime.path)
    with pytest.raises(ProviderError, match="Local model loading failed") as failure:
        complete(provider)
    assert failure.value.request_started is False
    assert runtime.model_loads == []
    assert provider.calls[0]["request_started"] is False


def test_context_reserves_output_and_fails_before_generation_without_truncation(
    runtime,
):
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", max_context_tokens=10
    )
    with pytest.raises(ProviderError, match="no truncation") as failure:
        complete(provider, max_output_tokens=8)
    assert failure.value.fatal is False
    assert failure.value.request_started is False
    assert runtime.model.generations == []
    assert runtime.tokenizer.templates[0][1].get("truncation") is None
    assert provider.context_limit == 10
    assert provider.calls[0]["input_tokens"] == 3


def test_user_context_limit_cannot_extend_native_config(runtime):
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", max_context_tokens=1000
    )
    complete(provider)
    assert provider.native_context_tokens == provider.context_limit == 64


def test_unknown_native_context_is_rejected(runtime):
    runtime.model.config = SimpleNamespace()
    provider = LocalTransformersProvider(runtime.path, device="cpu")
    with pytest.raises(ProviderError, match="Local model loading failed"):
        complete(provider)
    assert runtime.model.generations == []


def test_native_chat_template_is_required_without_fallback(runtime):
    runtime.tokenizer.chat_template = None
    provider = LocalTransformersProvider(runtime.path)
    with pytest.raises(ProviderError, match="Local tokenizer loading failed"):
        provider.count_tokens("text")
    assert runtime.model_loads == []


def test_incomplete_generation_records_actual_usage_and_is_never_retried(runtime):
    runtime.model.new_ids = [30] * 8
    provider = LocalTransformersProvider(runtime.path, device="cpu")
    with pytest.raises(ProviderError, match="truncated") as failure:
        complete(provider)
    assert failure.value.request_started is True
    assert failure.value.fatal is False
    assert len(runtime.model.generations) == 1
    assert provider.calls[0]["status"] == "incomplete"
    assert provider.calls[0]["usage"]["output_tokens"] == 8


def test_failure_metadata_omits_workflow_text_model_output_paths_and_exception_details(
    runtime,
):
    runtime.model.failure = RuntimeError(
        "PRIVATE_EXCEPTION_TEXT sk-private-token " + str(runtime.path)
    )
    provider = LocalTransformersProvider(runtime.path, device="cpu")
    with pytest.raises(ProviderError) as failure:
        complete(provider)
    encoded = json.dumps(provider.calls)
    assert "PUBLIC_WORKFLOW_TEXT" not in encoded
    assert "PRIVATE_EXCEPTION_TEXT" not in encoded
    assert "sk-private-token" not in encoded
    assert str(runtime.path) not in encoded
    assert len(str(failure.value)) < 200
    assert len(runtime.model.generations) == 1


def test_calls_are_defensive_and_local_hardware_cost_remains_unpriced(runtime):
    provider = LocalTransformersProvider(runtime.path, device="cpu")
    complete(provider)
    call = provider.calls[0]
    assert call["api_charge_usd"] == 0.0
    assert call["hardware_cost_usd"] is None
    assert call["cost_scope"] == "local_compute_unpriced"
    provider.calls[0]["usage"]["input_tokens"] = 999
    assert provider.calls[0]["usage"]["input_tokens"] == 3


def test_keyboard_interrupt_is_not_caught_or_retried(runtime):
    runtime.model.failure = KeyboardInterrupt()
    provider = LocalTransformersProvider(runtime.path, device="cpu")
    with pytest.raises(KeyboardInterrupt):
        complete(provider)
    assert len(runtime.model.generations) == 1
    assert provider.calls[0]["request_started"] is True


def test_missing_local_directory_never_attempts_a_download(runtime):
    provider = LocalTransformersProvider(runtime.path / "missing")
    with pytest.raises(ProviderError, match="no download"):
        provider.count_tokens("text")
    assert runtime.tokenizer_loads == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "auto"},
        {"max_context_tokens": True},
        {"max_context_tokens": 0},
        {"model": "bad model"},
    ],
)
def test_constructor_rejects_invalid_configuration(tmp_path, kwargs):
    with pytest.raises(ValueError):
        LocalTransformersProvider(tmp_path, **kwargs)
