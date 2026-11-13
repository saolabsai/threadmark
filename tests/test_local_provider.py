"""Mocked local-provider tests require neither torch nor model downloads."""

from contextlib import nullcontext
from hashlib import sha256
import json
import logging
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
        self.config = SimpleNamespace(max_position_embeddings=64, model_type="llama")
        self.generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=0)
        self.device = None
        self.evaluating = False
        self.frozen = False
        self.generations = []
        self.new_ids = [21, 22, 2]
        self.failure = None
        self.modules = [
            (
                "model.layers.0.self_attn.q_proj",
                SimpleNamespace(
                    weight=SimpleNamespace(shape=(6, 4)),
                    in_features=4,
                    out_features=6,
                ),
            )
        ]

    def named_modules(self):
        return iter(self.modules)

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
    monkeypatch.setitem(sys.modules, "peft", None)
    provider = LocalTransformersProvider(tmp_path, adapter_path=tmp_path / "adapter")
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
        {"adapter_path": ""},
        {"adapter_path": True},
        {"structured_workflow": 1},
        {"max_workflow_value_chars": True},
        {"max_workflow_value_chars": 0},
    ],
)
def test_constructor_rejects_invalid_configuration(tmp_path, kwargs):
    with pytest.raises(ValueError):
        LocalTransformersProvider(tmp_path, **kwargs)


class FakeWeight:
    def __init__(self, shape, floating=True):
        self.shape, self.floating = shape, floating

    def is_floating_point(self):
        return self.floating


@pytest.fixture
def adapter(runtime, monkeypatch):
    path = runtime.path / "adapter"
    path.mkdir()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "base_model_name_or_path": "/opt/ml/input/data/training/model",
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj"],
        "bias": "none",
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"mock safetensors")
    (runtime.path / "config.json").write_text(
        '{"model_type":"llama"}', encoding="utf-8"
    )
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    state = {
        prefix + ".lora_A.weight": FakeWeight((2, 4)),
        prefix + ".lora_B.weight": FakeWeight((6, 2)),
    }
    values = SimpleNamespace(
        path=path,
        config=config,
        state=state,
        configs=[],
        loads=[],
        attachments=[],
        assignments=[],
        result=SimpleNamespace(
            missing_keys=["base_model.model.model.embed_tokens.weight"],
            unexpected_keys=[],
        ),
    )

    def load_config(**kwargs):
        values.configs.append(kwargs)
        return SimpleNamespace(**kwargs)

    def load_weights(*args, **kwargs):
        values.loads.append((args, kwargs))
        return values.state

    def attach(*args, **kwargs):
        values.attachments.append((args, kwargs))
        return runtime.model

    def assign(*args, **kwargs):
        values.assignments.append((args, kwargs))
        return values.result

    def forbidden(*args, **kwargs):
        raise AssertionError("No PEFT or Hub file resolver may be called")

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=load_config,
            get_peft_model=attach,
            set_peft_model_state_dict=assign,
            PeftModel=SimpleNamespace(from_pretrained=forbidden),
            AutoPeftModelForCausalLM=SimpleNamespace(from_pretrained=forbidden),
        ),
    )
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "safetensors.torch", SimpleNamespace(load_file=load_weights)
    )
    return values


def test_local_adapter_uses_explicit_base_and_in_memory_safetensors(runtime, adapter):
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", adapter_path=adapter.path
    )
    metadata = provider.load()
    assert runtime.model_loads[0][0] == (str(runtime.path.resolve()),)
    assert runtime.tokenizer_loads[0][0] == (str(runtime.path.resolve()),)
    assert adapter.loads == [
        ((str(adapter.path / "adapter_model.safetensors"),), {"device": "cpu"})
    ]
    assert adapter.configs[0]["base_model_name_or_path"] == str(runtime.path.resolve())
    assert adapter.configs[0]["inference_mode"] is True
    assert "revision" not in adapter.configs[0]
    assert adapter.assignments[0][1] == {
        "adapter_name": "default",
        "ignore_mismatched_sizes": False,
    }
    assert runtime.model.frozen and runtime.model.evaluating
    identity = metadata["adapter"]
    assert identity["adapter_id"].startswith("sha256:")
    assert identity["weights_sha256"] == sha256(b"mock safetensors").hexdigest()
    assert identity["base_model_type"] == "llama"
    assert "training_base_revision_unverified" in identity["compatibility"]
    complete(provider)
    assert provider.calls[0]["adapter"] == identity
    assert provider.calls[0]["adapter_requested"] is True
    assert len(adapter.loads) == len(adapter.assignments) == 1
    assert str(runtime.path) not in json.dumps(metadata)
    assert "/opt/ml" not in json.dumps(provider.calls)
    identity["weights_sha256"] = "changed"
    assert provider.load()["adapter"]["weights_sha256"] != "changed"


def test_base_only_identity_is_explicitly_unadapted(runtime):
    provider = LocalTransformersProvider(runtime.path, device="cpu")
    assert provider.load()["adapter"] is None
    complete(provider)
    assert provider.calls[0]["adapter"] is None
    assert provider.calls[0]["adapter_requested"] is False


def test_adapter_token_count_loads_neither_base_nor_adapter_weights(runtime, adapter):
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", adapter_path=adapter.path
    )
    assert provider.count_tokens("hello") == 5
    assert runtime.model_loads == adapter.loads == adapter.attachments == []


@pytest.mark.parametrize(
    "missing", ["adapter_config.json", "adapter_model.safetensors"]
)
def test_missing_adapter_files_fail_without_model_load_or_pickle_fallback(
    runtime, adapter, missing
):
    (adapter.path / missing).unlink()
    (adapter.path / "adapter_model.bin").write_bytes(b"must not load pickle")
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", adapter_path=adapter.path
    )
    with pytest.raises(ProviderError, match="Local model loading failed"):
        provider.load()
    assert runtime.model_loads == adapter.loads == adapter.attachments == []


@pytest.mark.parametrize(
    "change",
    [
        {"peft_type": "PREFIX_TUNING"},
        {"task_type": "SEQ_CLS"},
        {"model_type": "gpt2"},
        {"r": True},
        {"r": 0},
        {"lora_alpha": float("nan")},
        {"lora_dropout": 1.0},
        {"target_modules": "q_proj"},
        {"target_modules": ["q_proj", "q_proj"]},
        {"modules_to_save": ["lm_head"]},
        {"use_dora": True},
        {"bias": "all"},
        {"layer_replication": [[0, 1]]},
        {"new_variant_option": True},
        {"init_lora_weights": "loftq"},
    ],
)
def test_unsupported_adapter_configuration_fails_before_base_load(
    runtime, adapter, change
):
    (adapter.path / "adapter_config.json").write_text(
        json.dumps({**adapter.config, **change}), encoding="utf-8"
    )
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", adapter_path=adapter.path
    )
    with pytest.raises(ProviderError, match="Local model loading failed"):
        provider.load()
    assert runtime.model_loads == adapter.loads == adapter.attachments == []


@pytest.mark.parametrize(
    "problem",
    ["missing", "extra", "shape", "integer", "target", "model_type", "nonlinear"],
)
def test_adapter_must_exactly_match_base_target_tensors(runtime, adapter, problem):
    key = next(iter(adapter.state))
    if problem == "missing":
        adapter.state.pop(key)
    elif problem == "extra":
        adapter.state["unrelated.weight"] = FakeWeight((2, 4))
    elif problem == "shape":
        adapter.state[key] = FakeWeight((2, 5))
    elif problem == "integer":
        adapter.state[key] = FakeWeight((2, 4), floating=False)
    elif problem == "target":
        runtime.model.modules = []
    elif problem == "nonlinear":
        del runtime.model.modules[0][1].in_features
    else:
        runtime.model.config.model_type = "gpt2"
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", adapter_path=adapter.path
    )
    with pytest.raises(ProviderError, match="Local model loading failed"):
        provider.load()
    assert adapter.attachments == []
    assert runtime.model.generations == []


@pytest.mark.parametrize("problem", ["missing", "extra"])
def test_peft_load_must_not_leave_uninitialized_adapter_tensors(
    runtime, adapter, problem
):
    if problem == "missing":
        adapter.result.missing_keys.append(
            "base_model.model.q_proj.lora_A.default.weight"
        )
    else:
        adapter.result.unexpected_keys.append("unexpected.weight")
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", adapter_path=adapter.path
    )
    with pytest.raises(ProviderError, match="Local model loading failed"):
        provider.load()
    assert provider._model is None
    assert runtime.model.generations == []


def test_adapter_identity_detects_changed_checkpoint(runtime, adapter):
    first = LocalTransformersProvider(
        runtime.path, device="cpu", adapter_path=adapter.path
    ).load()["adapter"]
    (adapter.path / "adapter_model.safetensors").write_bytes(
        b"different mock checkpoint"
    )
    second = LocalTransformersProvider(
        runtime.path, device="cpu", adapter_path=adapter.path
    ).load()["adapter"]
    assert first["adapter_id"] != second["adapter_id"]


def test_adapter_preserves_missing_eos_rejection_and_records_identity(runtime, adapter):
    runtime.model.new_ids = [30] * 8
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", adapter_path=adapter.path
    )
    with pytest.raises(ProviderError, match="truncated"):
        complete(provider)
    assert provider.calls[0]["status"] == "incomplete"
    assert (
        provider.calls[0]["adapter"]["weights_sha256"]
        == sha256(b"mock safetensors").hexdigest()
    )
    assert len(runtime.model.generations) == 1


def test_model_path_cannot_auto_resolve_an_adapter_base(runtime, adapter):
    provider = LocalTransformersProvider(adapter.path, device="cpu")
    with pytest.raises(ProviderError, match="base model"):
        provider.load()
    assert runtime.tokenizer_loads == runtime.model_loads == adapter.loads == []


@pytest.fixture
def grammar(monkeypatch, runtime):
    values = SimpleNamespace(data_builds=[], parsers=[], prefixes=[], invocations=[])

    def tokenizer_data(tokenizer):
        values.data_builds.append(tokenizer)
        return SimpleNamespace(tokenizer_alphabet='abc{}[]"', eos_token_id=-1)

    def parser(schema, *, config):
        result = SimpleNamespace(schema=schema, config=config)
        values.parsers.append(result)
        return result

    def prefix(data, instance):
        # The pinned package overwrites parser config during construction.
        instance.config = SimpleNamespace(force_json_field_order=True)

        def allowed(batch_id, ids):
            values.invocations.append((batch_id, ids))
            return [2]

        values.prefixes.append((data, instance, allowed))
        return allowed

    monkeypatch.setitem(
        sys.modules, "lmformatenforcer", SimpleNamespace(JsonSchemaParser=parser)
    )
    monkeypatch.setitem(
        sys.modules,
        "lmformatenforcer.characterlevelparser",
        SimpleNamespace(
            CharacterLevelParserConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setitem(sys.modules, "lmformatenforcer.integrations", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "lmformatenforcer.integrations.transformers",
        SimpleNamespace(
            build_token_enforcer_tokenizer_data=tokenizer_data,
            build_transformers_prefix_allowed_tokens_fn=prefix,
        ),
    )
    monkeypatch.setattr(
        "threadmark.local_provider.package_version", lambda name: "0.11.3"
    )
    runtime.tokenizer.output_text = '{"events":[]}'
    return values


def workflow_content(identifier="e1"):
    return {
        "entity_catalog": [{"entity_id": "p", "label": "Project", "fields": ["owner"]}],
        "reference_context": {"current_state": {"facts": []}, "open_proposals": []},
        "new_events": [
            {
                "event_id": identifier,
                "timestamp": "2026-11-09T00:00:00Z",
                "sender": "User",
                "text": "A workflow event.",
            }
        ],
    }


def workflow_complete(provider, **kwargs):
    return provider.complete(
        instructions="Extract workflow changes.",
        content=kwargs.pop("content", workflow_content()),
        max_output_tokens=kwargs.pop("max_output_tokens", 8),
        **kwargs,
    )


def test_structured_workflow_builds_fresh_schema_and_caches_only_tokenizer(
    runtime, grammar
):
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", structured_workflow=True
    )
    workflow_complete(provider)
    workflow_complete(provider, content=workflow_content("e2"))
    assert len(grammar.data_builds) == 1
    assert len(grammar.parsers) == len(grammar.prefixes) == 2
    assert grammar.parsers[0] is not grammar.parsers[1]
    assert grammar.prefixes[0][0].eos_token_id == [2]
    for instance in grammar.parsers:
        assert instance.config.force_json_field_order is False
        assert instance.config.max_consecutive_whitespaces == 12
        assert instance.config.max_json_array_length == 128
    call1, call2 = provider.calls
    assert call1["structured_workflow"] is call2["structured_workflow"] is True
    assert call1["workflow_schema_sha256"] != call2["workflow_schema_sha256"]
    assert call1["constraint_engine"] == "lm-format-enforcer/0.11.3"
    assert call1["constraint_setup_seconds"] >= 0
    prefix = runtime.model.generations[0]["prefix_allowed_tokens_fn"]
    assert prefix(0, FakeVector([11])) == [2]
    assert len(grammar.invocations) == 1
    assert provider.load()["structured_workflow"] is True


def test_structured_mode_can_toggle_sequentially_without_schema_state_leak(
    runtime, grammar
):
    provider = LocalTransformersProvider(runtime.path, device="cpu")
    workflow_complete(provider)
    provider.structured_workflow = True
    workflow_complete(provider)
    provider.structured_workflow = False
    workflow_complete(provider)
    assert [r["structured_workflow"] for r in provider.calls] == [False, True, False]
    assert [bool(r["workflow_schema_sha256"]) for r in provider.calls] == [
        False,
        True,
        False,
    ]
    assert ["prefix_allowed_tokens_fn" in g for g in runtime.model.generations] == [
        False,
        True,
        False,
    ]
    assert len(runtime.model_loads) == 1


def test_structured_mode_rejects_reader_or_malformed_requests_before_model_load(
    runtime, grammar
):
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", structured_workflow=True
    )
    with pytest.raises(ProviderError, match="request validation") as failure:
        complete(provider)
    assert failure.value.request_started is False
    assert runtime.model_loads == runtime.model.generations == grammar.parsers == []
    assert provider.calls[0]["workflow_schema_sha256"] is None


@pytest.mark.parametrize(
    "text",
    [
        '{"events":[],"events":[]}',
        '{"events":[],"extra":1}',
        '{"events":[{"event_id":"foreign","operations":[]}]}',
        '{"events":[{"event_id":"e1","operations":[{"op":"delete",'
        '"entity_id":"p","field":"owner","value":null,'
        '"proposal_id":null,"reason":null}]}]}',
        '{"events":[{"event_id":"e1","operations":[{"op":"set",'
        '"entity_id":"p","field":"owner","value":null,'
        '"proposal_id":null,"reason":null}]}]}',
        "not-json",
    ],
)
def test_structured_postvalidation_rejects_even_native_eos_after_constraint_failure(
    runtime, grammar, text
):
    runtime.tokenizer.output_text = text
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", structured_workflow=True
    )
    with pytest.raises(ProviderError, match="strict validation") as failure:
        workflow_complete(provider)
    assert failure.value.request_started is True
    assert provider.calls[0]["status"] == "invalid_structured_output"
    assert provider.calls[0]["usage"]["output_tokens"] == 3
    assert len(runtime.model.generations) == 1


def test_structured_mode_does_not_silently_repair_semantically_wrong_values(
    runtime, grammar
):
    runtime.tokenizer.output_text = json.dumps(
        {
            "events": [
                {
                    "event_id": "e1",
                    "operations": [
                        {
                            "op": "set",
                            "entity_id": "p",
                            "field": "owner",
                            "value": "unmentioned",
                            "proposal_id": None,
                            "reason": None,
                        }
                    ],
                }
            ]
        }
    )
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", structured_workflow=True
    )
    assert workflow_complete(provider).text == runtime.tokenizer.output_text


def test_structured_mode_respects_configured_value_length(runtime, grammar):
    runtime.tokenizer.output_text = json.dumps(
        {
            "events": [
                {
                    "event_id": "e1",
                    "operations": [
                        {
                            "op": "set",
                            "entity_id": "p",
                            "field": "owner",
                            "value": "long",
                            "proposal_id": None,
                            "reason": None,
                        }
                    ],
                }
            ]
        }
    )
    provider = LocalTransformersProvider(
        runtime.path,
        device="cpu",
        structured_workflow=True,
        max_workflow_value_chars=3,
    )
    with pytest.raises(ProviderError, match="strict validation"):
        workflow_complete(provider)


def test_structured_mode_keeps_eos_rejection(runtime, grammar):
    runtime.model.new_ids = [30] * 8
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", structured_workflow=True
    )
    with pytest.raises(ProviderError, match="truncated"):
        workflow_complete(provider)
    assert provider.calls[0]["status"] == "incomplete"


def test_missing_constraint_dependency_has_no_unconstrained_fallback(
    runtime, grammar, monkeypatch
):
    monkeypatch.setitem(sys.modules, "lmformatenforcer", None)
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", structured_workflow=True
    )
    with pytest.raises(ProviderError, match="constraint initialization") as failure:
        workflow_complete(provider)
    assert failure.value.request_started is False
    assert runtime.model.generations == []


def test_constraint_runtime_version_must_match_recorded_pin(
    runtime, grammar, monkeypatch
):
    monkeypatch.setattr(
        "threadmark.local_provider.package_version", lambda name: "99.0"
    )
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", structured_workflow=True
    )
    with pytest.raises(ProviderError, match="constraint initialization"):
        workflow_complete(provider)
    assert runtime.model.generations == []


def test_constraint_internal_prefix_logs_are_redacted_and_abort_without_retry(
    runtime, grammar, monkeypatch, caplog
):
    secret = "PRIVATE_WORKFLOW_PREFIX"
    original_generate = runtime.model.generate
    root_logger = logging.getLogger()
    initial_filters = list(root_logger.filters)

    def logging_prefix(data, parser):
        def invoke(batch_id, ids):
            root_logger.warning("Unrelated application diagnostic")
            record = logging.LogRecord(
                name="root",
                level=logging.ERROR,
                pathname="C:/packages/lmformatenforcer/tokenenforcer.py",
                lineno=1,
                msg="Parser prefix: %s",
                args=(secret,),
                exc_info=None,
            )
            record.stack_info = secret
            record.exc_text = secret
            root_logger.handle(record)
            return [2]  # The library's unsafe EOS-on-error behavior is rejected.

        return invoke

    monkeypatch.setattr(
        sys.modules["lmformatenforcer.integrations.transformers"],
        "build_transformers_prefix_allowed_tokens_fn",
        logging_prefix,
    )

    def generate(**kwargs):
        kwargs["prefix_allowed_tokens_fn"](0, FakeVector([11]))
        return original_generate(**kwargs)

    runtime.model.generate = generate
    provider = LocalTransformersProvider(
        runtime.path, device="cpu", structured_workflow=True
    )
    with pytest.raises(ProviderError, match="constraint engine failed") as failure:
        workflow_complete(provider)
    assert failure.value.request_started is True
    assert secret not in caplog.text
    assert secret not in json.dumps(provider.calls)
    assert "Unrelated application diagnostic" in caplog.text
    assert "redacted" in caplog.text
    assert list(root_logger.filters) == initial_filters
    assert runtime.model.generations == []
