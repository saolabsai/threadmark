"""Optional local dense retrieval with complete-event chunk coverage.

The intended encoder is sentence-transformers/all-MiniLM-L6-v2. It is loaded
through AutoModel from existing local safetensors files; this module downloads
nothing. Masked mean pooling and L2 normalization produce one vector per native
token chunk. The maximum query-chunk/event-chunk cosine ranks each event.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from .config import Settings
from .context import TokenCounter
from .providers import ProviderError
from .schema import canonical, catalog_keys, public_catalog, public_events, public_query


def _unit_vector(values: list[float]) -> tuple[float, ...]:
    if not values or any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("Encoder returned an invalid embedding")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("Encoder returned a zero or invalid embedding")
    return tuple(value / norm for value in values)


def _expanded_query(query: dict[str, Any], catalog: list[dict[str, Any]]) -> str:
    entities = {entity["entity_id"]: entity for entity in catalog}
    parts = [query["text"]]
    for requested in query["requested_facts"]:
        entity = entities[requested["entity_id"]]
        parts.append(
            "Entity: "
            + " / ".join(
                entity[key] for key in ("entity_id", "label", "kind") if key in entity
            )
        )
        field = requested["field"]
        parts.append("Field: " + field)
        definition = entity.get("field_definitions", {}).get(field)
        if definition:
            parts.append("Definition: " + definition)
    return "\n".join(parts)


class LocalDenseRetriever:
    """Encode all original content locally and select whole original events.

    Ranking is the maximum cosine over every query/event chunk pair. Native
    token chunks cover the full text, including event endings. This avoids
    silently discarding text, but cross-chunk relations and workflow authority
    remain limitations. Recent original events supplement positive ranked hits.

    The most recent archive's embeddings are cached in memory across questions.
    Its first selection includes indexing work. Model initialization can be
    measured separately with ``load``. No predicted memory or gold labels enter
    either embeddings or ranking. One instance supports sequential use.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        if not isinstance(model_path, (str, Path)) or not str(model_path).strip():
            raise ValueError("model_path must identify a local encoder directory")
        if not isinstance(device, str) or not re.fullmatch(
            r"cpu|cuda(?::[0-9]+)?", device
        ):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self.native_context_tokens: int | None = None
        self.embedding_context_tokens: int | None = None
        self.content_chunk_tokens: int | None = None
        self.load_seconds = 0.0
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._cached_archive: tuple[str, ...] | None = None
        self._cached_vectors: list[list[tuple[float, ...]]] = []
        self._last_selection: dict[str, Any] = {}

    @property
    def last_selection(self) -> dict[str, Any]:
        return deepcopy(self._last_selection)

    def load(self) -> dict[str, Any]:
        """Load an existing local encoder once, without downloads or API calls."""
        already_loaded = self._model is not None
        self._ensure_model()
        return {
            "method": "dense_max_chunk_cosine_v1",
            "device": self.device,
            "dtype": "float32",
            "pooling": "attention_masked_mean_then_l2_normalize",
            "event_score": "maximum_cosine_over_all_query_and_event_chunks",
            "native_context_tokens": self.native_context_tokens,
            "embedding_context_tokens": self.embedding_context_tokens,
            "content_chunk_tokens": self.content_chunk_tokens,
            "load_seconds": self.load_seconds,
            "already_loaded": already_loaded,
            "api_charge_usd": 0.0,
            "hardware_cost_usd": None,
            "cost_scope": "local_compute_unpriced",
        }

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise ProviderError(
                "Local encoder directory is unavailable; no download or retry was made"
            )
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable")
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=False,
                token=False,
            )
            model = AutoModel.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                token=False,
                torch_dtype=torch.float32,
            )
            native = getattr(model.config, "max_position_embeddings", None)
            if type(native) is not int or not 1 <= native <= 1_000_000:
                raise ValueError("Encoder config must declare a native context limit")
            limits = [native]
            tokenizer_limit = getattr(tokenizer, "model_max_length", None)
            if type(tokenizer_limit) is int and 1 <= tokenizer_limit <= 1_000_000:
                limits.append(tokenizer_limit)
            # Sentence-transformers can specify a smaller intended encoding
            # window than the underlying transformer positional capacity.
            sentence_config = self.model_path / "sentence_bert_config.json"
            if sentence_config.exists():
                if sentence_config.stat().st_size > 64_000:
                    raise ValueError("Sentence-transformer config is oversized")
                limit = json.loads(sentence_config.read_text(encoding="utf-8")).get(
                    "max_seq_length"
                )
                if type(limit) is not int or not 1 <= limit <= native:
                    raise ValueError("Invalid sentence-transformer encoding window")
                limits.append(limit)
            effective = min(limits)
            special_count = tokenizer.num_special_tokens_to_add(pair=False)
            if type(special_count) is not int or not 0 <= special_count < effective:
                raise ValueError("Invalid tokenizer special-token reservation")
            model.to(self.device)
            model.eval()
            model.requires_grad_(False)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
        except Exception as exc:
            raise ProviderError(
                f"Local encoder loading failed ({type(exc).__name__}); no download or retry was made"
            ) from None
        self._tokenizer, self._model, self._torch = tokenizer, model, torch
        self.native_context_tokens = native
        self.embedding_context_tokens = effective
        self.content_chunk_tokens = effective - special_count
        self.load_seconds = time.perf_counter() - started

    def _embed_chunks(self, text: str) -> list[tuple[float, ...]]:
        self._ensure_model()
        try:
            token_ids = self._tokenizer.encode(
                text, add_special_tokens=False, truncation=False
            )
            if not isinstance(token_ids, list) or any(
                type(value) is not int or value < 0 for value in token_ids
            ):
                raise ValueError("Tokenizer returned invalid token IDs")
            size = self.content_chunk_tokens
            chunks = [
                token_ids[offset : offset + size]
                for offset in range(0, len(token_ids), size)
            ] or [[]]
            vectors = []
            for chunk in chunks:
                batch = self._tokenizer.prepare_for_model(
                    chunk,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                    return_attention_mask=True,
                    return_tensors="pt",
                    prepend_batch_axis=True,
                )
                input_ids = batch["input_ids"]
                special_count = (
                    self.embedding_context_tokens - self.content_chunk_tokens
                )
                if (
                    input_ids.ndim != 2
                    or input_ids.shape[0] != 1
                    or input_ids.shape[1] != len(chunk) + special_count
                    or input_ids.shape[1] > self.embedding_context_tokens
                ):
                    raise ValueError(
                        "Tokenizer exceeded the reserved native chunk window"
                    )
                inputs = {
                    key: batch[key].to(self.device)
                    for key in ("input_ids", "attention_mask", "token_type_ids")
                    if key in batch
                }
                if "attention_mask" not in inputs:
                    raise ValueError("Encoder requires an attention mask")
                with self._torch.inference_mode():
                    hidden = self._model(**inputs, return_dict=True).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                        min=1e-9
                    )
                values = pooled[0].detach().cpu().tolist()
                vectors.append(_unit_vector(values))
            return vectors
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Local encoder embedding failed ({type(exc).__name__}); no retry was made"
            ) from None

    def select(
        self,
        events: list[dict[str, Any]],
        query: dict[str, Any],
        catalog: list[dict[str, Any]],
        count_tokens: TokenCounter,
        settings: Settings,
        top_k: int = 8,
    ) -> list[dict[str, str]]:
        """Return positive dense hits plus the same recent whole-event suffix.

        Original events are deduplicated chronologically. The union must fit
        ``max_source_bytes``; oversized packets fail without text truncation.
        ``count_tokens`` controls the recent window, independently of the local
        encoder's native tokenizer used to cover complete event/query text.
        """
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if not callable(count_tokens):
            raise TypeError("count_tokens must be callable")
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        catalog = public_catalog(catalog)
        query = public_query(query, catalog_keys(catalog))
        events = public_events(events)
        started = time.perf_counter()
        self._last_selection = {}
        documents = [event["text"] + "\nSender: " + event["sender"] for event in events]
        fingerprints = tuple(
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in documents
        )
        cache_hit = fingerprints == self._cached_archive
        indexing_started = time.perf_counter()
        if not cache_hit:
            vectors = [self._embed_chunks(text) for text in documents]
            self._cached_archive, self._cached_vectors = fingerprints, vectors
        indexing_seconds = time.perf_counter() - indexing_started
        query_vectors = (
            self._embed_chunks(_expanded_query(query, catalog)) if events else []
        )
        scores = []
        for index, chunks in enumerate(self._cached_vectors):
            if any(
                len(document) != len(question)
                for document in chunks
                for question in query_vectors
            ):
                raise ValueError("Encoder embedding dimensions disagree")
            score = max(
                sum(left * right for left, right in zip(document, question))
                for document in chunks
                for question in query_vectors
            )
            if score > 0:
                scores.append((score, index))
        selected = {index for _, index in sorted(scores, reverse=True)[:top_k]}
        recent: list[dict[str, str]] = []
        for index in range(len(events) - 1, -1, -1):
            trial = [events[index], *recent]
            count = count_tokens(canonical({"source_events": trial}))
            if type(count) is not int or count < 0:
                raise ValueError("Token counter must return a nonnegative integer")
            if count > settings.recent_source_tokens:
                break
            recent = trial
            selected.add(index)
        sources = [event for index, event in enumerate(events) if index in selected]
        if (
            len(canonical({"source_events": sources}).encode("utf-8"))
            > settings.max_source_bytes
        ):
            raise ValueError("Selected dense sources exceed the bounded byte size")
        self._last_selection = {
            "method": "dense_max_chunk_cosine_v1",
            "archive_events": len(events),
            "archive_native_chunks": sum(
                len(chunks) for chunks in self._cached_vectors
            ),
            "query_native_chunks": len(query_vectors),
            "index_cache_hit": cache_hit,
            "indexing_seconds": indexing_seconds,
            "selection_seconds": time.perf_counter() - started,
            "selected_events": len(sources),
        }
        return sources
