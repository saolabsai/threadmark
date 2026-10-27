"""Dense ranking and native chunking tests without model dependencies."""

from contextlib import nullcontext
from copy import deepcopy
import json
import math
import sys
from types import SimpleNamespace

import pytest

from threadmark import Settings
from threadmark.dense_retrieval import LocalDenseRetriever, _unit_vector
from threadmark.providers import ProviderError
from threadmark.schema import canonical

CATALOG = [
    {
        "entity_id": "alpha",
        "label": "Alpha review",
        "fields": ["owner"],
        "field_definitions": {"owner": "Current facilitator name."},
    },
    {"entity_id": "beta", "label": "Beta review", "fields": ["owner"]},
]
QUERY = {
    "text": "Who is responsible?",
    "requested_facts": [{"entity_id": "alpha", "field": "owner"}],
}


def event(identifier, minute, text):
    return {
        "event_id": identifier,
        "timestamp": f"2026-10-28T10:{minute:02}:00+00:00",
        "sender": "Recorder",
        "text": text,
    }


@pytest.fixture
def mocked_retriever(tmp_path, monkeypatch):
    retriever = LocalDenseRetriever(tmp_path)
    calls = []

    def embed(text):
        calls.append(text)
        if text.startswith(QUERY["text"]):
            return [(1.0, 0.0)]
        if text.startswith("relevant"):
            return [(1.0, 0.0)]
        if text.startswith("partial"):
            return [(0.6, 0.8)]
        if text.startswith("tail-match"):
            return [(0.0, 1.0), (1.0, 0.0)]
        return [(-1.0, 0.0)]

    monkeypatch.setattr(retriever, "_embed_chunks", embed)
    return retriever, calls


def select(retriever, events, *, query=QUERY, catalog=CATALOG, top_k=1, settings=None):
    return retriever.select(
        events, query, catalog, len, settings or Settings(recent_source_tokens=1), top_k
    )


def test_positive_dense_ranking_finds_relevant_old_event(mocked_retriever):
    retriever, _ = mocked_retriever
    events = [
        event("e1", 0, "relevant owner assignment"),
        event("e2", 1, "partial related detail"),
        event("e3", 2, "irrelevant background"),
    ]
    assert [source["event_id"] for source in select(retriever, events)] == ["e1"]


def test_maximum_chunk_cosine_retains_match_at_event_ending(mocked_retriever):
    retriever, _ = mocked_retriever
    events = [
        event("e1", 0, "partial related detail"),
        event("e2", 1, "tail-match is after long context"),
    ]
    assert [source["event_id"] for source in select(retriever, events)] == ["e2"]


def test_recent_clearing_is_included_even_with_negative_similarity(mocked_retriever):
    retriever, _ = mocked_retriever
    events = [
        event("e1", 0, "relevant owner assignment"),
        event("e2", 1, "irrelevant background"),
        event("e3", 2, "That assignment is now cleared."),
    ]
    cap = len(canonical({"source_events": [events[-1]]}))
    assert [
        source["event_id"]
        for source in select(
            retriever, events, settings=Settings(recent_source_tokens=cap)
        )
    ] == ["e1", "e3"]


def test_expansion_uses_requested_public_fields_and_gold_never_enters_embeddings(
    mocked_retriever,
):
    retriever, calls = mocked_retriever
    events = [event("e1", 0, "relevant owner assignment")]
    events[0]["gold_state"] = "PRIVATE_EVENT_GOLD"
    catalog = deepcopy(CATALOG)
    catalog[0]["expected"] = "PRIVATE_CATALOG_GOLD"
    query = {**QUERY, "expected": "PRIVATE_QUERY_GOLD"}
    sources = select(retriever, events, query=query, catalog=catalog)
    encoded = "\n".join(calls)
    assert "Alpha review" in encoded
    assert "Current facilitator name." in encoded
    assert "Beta review" not in encoded
    assert "PRIVATE_" not in encoded
    assert set(sources[0]) == {"event_id", "timestamp", "sender", "text"}


def test_archive_vectors_are_cached_but_changed_content_is_reindexed(mocked_retriever):
    retriever, calls = mocked_retriever
    events = [
        event("e1", 0, "relevant owner assignment"),
        event("e2", 1, "partial related detail"),
    ]
    select(retriever, events)
    assert retriever.last_selection["index_cache_hit"] is False
    select(retriever, events)
    assert retriever.last_selection["index_cache_hit"] is True
    assert len(calls) == 4
    events[0]["text"] = "partial changed content"
    select(retriever, events)
    assert retriever.last_selection["index_cache_hit"] is False
    assert len(calls) == 7


def test_selection_deduplicates_chronologically_and_returns_defensive_copies(
    mocked_retriever,
):
    retriever, _ = mocked_retriever
    events = [event("e1", 0, "relevant older"), event("e2", 1, "relevant newer")]
    sources = select(retriever, events, top_k=2, settings=Settings())
    assert [source["event_id"] for source in sources] == ["e1", "e2"]
    sources[0]["text"] = "mutated"
    assert events[0]["text"] == "relevant older"
    retriever.last_selection["selected_events"] = 99
    assert retriever.last_selection["selected_events"] == 2


def test_equal_dense_scores_prefer_latest_and_negative_scores_are_excluded(
    mocked_retriever,
):
    retriever, _ = mocked_retriever
    events = [event("e1", 0, "relevant older"), event("e2", 1, "relevant newer")]
    assert [source["event_id"] for source in select(retriever, events)] == ["e2"]
    assert select(retriever, [event("e1", 0, "negative-only background")]) == []


def test_source_byte_cap_fails_without_silently_dropping_a_hit(mocked_retriever):
    retriever, _ = mocked_retriever
    events = [event("e1", 0, "relevant owner assignment")]
    exact = len(canonical({"source_events": events}).encode("utf-8"))
    assert (
        select(retriever, events, settings=Settings(max_source_bytes=exact)) == events
    )
    with pytest.raises(ValueError, match="bounded byte"):
        select(retriever, events, settings=Settings(max_source_bytes=exact - 1))


def test_empty_archive_needs_no_encoder_and_resets_previous_cache(mocked_retriever):
    retriever, calls = mocked_retriever
    select(retriever, [event("e1", 0, "relevant")])
    calls.clear()
    assert select(retriever, []) == []
    assert calls == []
    assert retriever.last_selection["archive_native_chunks"] == 0


@pytest.mark.parametrize("top_k", [0, True, 1.5, 101])
def test_top_k_validation(mocked_retriever, top_k):
    retriever, _ = mocked_retriever
    with pytest.raises(ValueError, match="top_k"):
        select(retriever, [], top_k=top_k)


class Tensor:
    """Minimal tensor operations used by masked mean pooling in these tests."""

    dtype = "float32"

    def __init__(self, values):
        self.values = values
        shape, part = [], values
        while isinstance(part, list):
            shape.append(len(part))
            part = part[0] if part else 0
        self.shape = tuple(shape)
        self.ndim = len(shape)

    def to(self, _):
        return self

    def unsqueeze(self, dimension):
        assert dimension == -1 and self.ndim == 2
        return Tensor([[[value] for value in row] for row in self.values])

    def __mul__(self, other):
        return Tensor(
            [
                [
                    [value * mask[0] for value in vector]
                    for vector, mask in zip(vectors, masks)
                ]
                for vectors, masks in zip(self.values, other.values)
            ]
        )

    def sum(self, dim):
        assert dim == 1 and self.ndim == 3
        return Tensor(
            [
                [sum(vector[index] for vector in row) for index in range(len(row[0]))]
                for row in self.values
            ]
        )

    def clamp(self, min):
        return Tensor([[max(value, min) for value in row] for row in self.values])

    def __truediv__(self, other):
        return Tensor(
            [
                [value / divisor[0] for value in row]
                for row, divisor in zip(self.values, other.values)
            ]
        )

    def __getitem__(self, item):
        return Tensor(self.values[item])

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return deepcopy(self.values)


@pytest.fixture
def native_runtime(monkeypatch, tmp_path):
    prepared, tokenizer_loads, model_loads = [], [], []

    class Tokenizer:
        model_max_length = 6

        def num_special_tokens_to_add(self, pair):
            assert pair is False
            return 2

        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return [ord(character) for character in text]

        def prepare_for_model(self, ids, **kwargs):
            prepared.append(ids[:])
            assert kwargs["truncation"] is False
            # HF prepare_for_model defaults to a 1D token tensor, whereas the
            # encoder requires an explicit batch dimension for a single chunk.
            assert kwargs["prepend_batch_axis"] is True
            values = [101] + ids + [102]
            return {
                "input_ids": Tensor([values]),
                "attention_mask": Tensor([[1] * (len(values) - 1) + [0]]),
            }

    class Model:
        config = SimpleNamespace(max_position_embeddings=8)
        frozen = False
        evaluating = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluating = True
            return self

        def requires_grad_(self, value):
            self.frozen = not value

        def __call__(self, **kwargs):
            ids = kwargs["input_ids"].values[0]
            vectors = [[2.0, 0.0]] + [[0.0, 2.0] for _ in ids[1:-1]] + [[999.0, 999.0]]
            return SimpleNamespace(last_hidden_state=Tensor([vectors]))

    tokenizer, model = Tokenizer(), Model()

    def tokenizer_loader(*args, **kwargs):
        tokenizer_loads.append(kwargs)
        return tokenizer

    def model_loader(*args, **kwargs):
        model_loads.append(kwargs)
        return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModel=SimpleNamespace(from_pretrained=model_loader),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="float32",
            inference_mode=lambda: nullcontext(),
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    return SimpleNamespace(
        path=tmp_path,
        tokenizer=tokenizer,
        model=model,
        prepared=prepared,
        tokenizer_loads=tokenizer_loads,
        model_loads=model_loads,
    )


def test_constructor_is_lazy_without_optional_dependencies(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    retriever = LocalDenseRetriever(tmp_path)
    assert retriever.device == "cpu"
    assert retriever.last_selection == {}


def test_local_safetensors_load_is_frozen_and_initialization_is_separate(
    native_runtime,
):
    retriever = LocalDenseRetriever(native_runtime.path)
    metadata = retriever.load()
    assert metadata["already_loaded"] is False
    assert metadata["native_context_tokens"] == 8
    assert metadata["embedding_context_tokens"] == 6
    assert metadata["content_chunk_tokens"] == 4
    assert metadata["hardware_cost_usd"] is None
    assert retriever.load()["already_loaded"] is True
    assert len(native_runtime.model_loads) == 1
    assert native_runtime.model.frozen and native_runtime.model.evaluating
    assert native_runtime.tokenizer_loads[0] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "token": False,
    }
    assert native_runtime.model_loads[0] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "token": False,
        "torch_dtype": "float32",
    }


def test_contiguous_native_chunks_cover_every_event_token_without_truncation(
    native_runtime,
):
    retriever = LocalDenseRetriever(native_runtime.path)
    vectors = retriever._embed_chunks("abcdefghij")
    assert native_runtime.prepared == [
        [ord(c) for c in "abcd"],
        [ord(c) for c in "efgh"],
        [ord(c) for c in "ij"],
    ]
    assert len(vectors) == 3
    assert all(
        sum(value * value for value in vector) == pytest.approx(1.0)
        for vector in vectors
    )


def test_attention_masked_mean_excludes_padded_hidden_vectors(native_runtime):
    retriever = LocalDenseRetriever(native_runtime.path)
    vector = retriever._embed_chunks("a")[0]
    assert vector == pytest.approx((1 / math.sqrt(2), 1 / math.sqrt(2)))


def test_sentence_transformer_smaller_window_is_honored(native_runtime):
    (native_runtime.path / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": 4}), encoding="utf-8"
    )
    retriever = LocalDenseRetriever(native_runtime.path)
    metadata = retriever.load()
    assert metadata["embedding_context_tokens"] == 4
    assert metadata["content_chunk_tokens"] == 2
    retriever._embed_chunks("abcdef")
    assert native_runtime.prepared == [
        [ord(c) for c in "ab"],
        [ord(c) for c in "cd"],
        [ord(c) for c in "ef"],
    ]


def test_long_query_is_chunked_with_all_public_expansion_content(native_runtime):
    retriever = LocalDenseRetriever(native_runtime.path)
    source = event("e1", 0, "brief")
    retriever.select([source], QUERY, CATALOG, len, Settings(recent_source_tokens=1))
    assert retriever.last_selection["query_native_chunks"] > 1
    assert retriever.last_selection["archive_native_chunks"] > 1


@pytest.mark.parametrize(
    "values", [[], [0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]]
)
def test_invalid_or_zero_embeddings_fail(values):
    with pytest.raises(ValueError, match="embedding"):
        _unit_vector(values)


def test_missing_local_assets_do_not_attempt_a_download(tmp_path):
    retriever = LocalDenseRetriever(tmp_path / "missing")
    with pytest.raises(ProviderError, match="no download"):
        retriever.load()
