"""Small, provider-independent workflow memory engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from .config import Settings
from .context import TokenCounter, chunks_for, select_source_events, token_counter
from .prompts import READER_INSTRUCTIONS, UPDATER_INSTRUCTIONS
from .retrieval import select_bm25_sources
from .schema import (
    MAX_ARCHIVE_BYTES,
    canonical,
    catalog_keys,
    initial_state,
    parse_json,
    public_catalog,
    public_events,
    public_query,
    validate_facts,
)


def _instructions(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 16_000:
        raise ValueError(
            f"{name} must be a nonempty string of at most 16000 characters"
        )
    return value


class Threadmark:
    """Maintain typed state and answer from selected original workflow evidence.

    A provider implements ``complete(instructions=..., content=...,
    max_output_tokens=...)`` and returns an object with a ``text`` attribute.
    Use one writer per instance; concurrent mutation is not supported. The full
    original event archive is retained. Schema validation checks structure and
    citation membership, not whether a model's factual interpretation is true.
    """

    def __init__(
        self,
        entity_catalog: list[dict[str, Any]],
        provider: Any,
        settings: Settings | None = None,
        count_tokens: TokenCounter | None = None,
        *,
        updater_instructions: str = UPDATER_INSTRUCTIONS,
        reader_instructions: str = READER_INSTRUCTIONS,
    ) -> None:
        self.settings = settings if settings is not None else Settings()
        if not isinstance(self.settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if not callable(getattr(provider, "complete", None)):
            raise TypeError("provider must implement complete")
        for attribute, expected in (
            ("model", self.settings.model),
            ("reasoning_effort", self.settings.reasoning_effort),
        ):
            if (
                hasattr(provider, attribute)
                and getattr(provider, attribute) != expected
            ):
                raise ValueError(f"Provider {attribute} must match Settings")
        if count_tokens is not None and not callable(count_tokens):
            raise TypeError("count_tokens must be callable")
        self._updater_instructions = _instructions(
            updater_instructions, "updater_instructions"
        )
        self._reader_instructions = _instructions(
            reader_instructions, "reader_instructions"
        )
        self._snapshot_version = (
            1
            if (
                updater_instructions == UPDATER_INSTRUCTIONS
                and reader_instructions == READER_INSTRUCTIONS
            )
            else 2
        )
        self._catalog = public_catalog(entity_catalog)
        self._keys = catalog_keys(self._catalog)
        self._provider = provider
        self._count_tokens = count_tokens
        self._events: list[dict[str, str]] = []
        self._baseline_only = False
        self._state = validate_facts(
            initial_state(self._keys), self._keys, set(), self.settings, memory=True
        )

    def _counter(self) -> TokenCounter:
        if self._count_tokens is None:
            self._count_tokens = token_counter()
        return self._count_tokens

    @property
    def state(self) -> dict[str, Any]:
        return deepcopy(self._state)

    @property
    def events(self) -> list[dict[str, str]]:
        return deepcopy(self._events)

    @property
    def entity_catalog(self) -> list[dict[str, Any]]:
        return deepcopy(self._catalog)

    def ingest(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Process chronological new events and return the complete current state.

        State and archive commit atomically only after every chunk validates.
        If a later chunk fails, prior provider calls may already have been
        charged. Nothing is retried automatically, and no paid fallback runs.
        For an archive loaded by ``from_events``, this first ingests the entire
        loaded archive plus the supplied new events; ``ingest([])`` ingests the
        loaded archive on its own.
        """
        new_events = public_events(events, prior=self._events)
        if not new_events and not self._baseline_only:
            return self.state
        pending = self._events + new_events if self._baseline_only else new_events
        chunks = chunks_for(pending, self._counter(), self.settings)
        candidate = (
            initial_state(self._keys) if self._baseline_only else deepcopy(self._state)
        )
        allowed_ids = (
            set()
            if self._baseline_only
            else {event["event_id"] for event in self._events}
        )
        for chunk in chunks:
            allowed_ids.update(event["event_id"] for event in chunk)
            completion = self._provider.complete(
                instructions=self._updater_instructions,
                content={
                    "entity_catalog": deepcopy(self._catalog),
                    "current_memory": deepcopy(candidate),
                    "new_events": deepcopy(chunk),
                },
                max_output_tokens=self.settings.updater_max_output_tokens,
            )
            candidate = validate_facts(
                completion.text, self._keys, allowed_ids, self.settings, memory=True
            )
        self._events = self._events + new_events
        self._state = candidate
        self._baseline_only = False
        return self.state

    def evidence(self, query: dict[str, Any]) -> list[dict[str, str]]:
        """Return the active reader's original source packet without a model call."""
        if self._baseline_only:
            raise ValueError(
                "Archive-only baseline has no ingested memory; call ingest([]) first"
            )
        query = public_query(query, self._keys)
        return deepcopy(
            select_source_events(
                self._events, query, self._state, self._counter(), self.settings
            )
        )

    def answer(
        self, query: dict[str, Any], mode: str = "active_memory"
    ) -> dict[str, Any]:
        """Answer from selected evidence, full history, BM25, or predicted state.

        Reader modes use exactly the same instructions and response limits.
        BM25 uses its default eight keyword-ranked whole events plus the recent
        suffix, and does not need memory ingestion or consult predicted facts.
        State mode directly returns unverified predicted current facts without
        retrieval or a provider call. The caller supplies requested entity/field
        pairs; this is explicit key routing, not free-form question answering.
        Unknown/unsupported values can be null. Citation validation only admits
        event IDs in the context actually supplied to this call.
        """
        query = public_query(query, self._keys)
        if mode == "state":
            if self._baseline_only:
                raise ValueError(
                    "Archive-only baseline has no ingested memory; call ingest([]) first"
                )
            keys = [
                (item["entity_id"], item["field"]) for item in query["requested_facts"]
            ]
            current = {
                (fact["entity_id"], fact["field"]): fact
                for fact in self._state["facts"]
            }
            return validate_facts(
                {"facts": [current[key] for key in keys]},
                keys,
                {event["event_id"] for event in self._events},
                self.settings,
            )
        if mode == "active_memory":
            sources = self.evidence(query)
            context = {"source_events": sources}
        elif mode == "full_history":
            sources = self.events
            context = {"events": sources}
        elif mode == "bm25":
            sources = select_bm25_sources(
                self._events, query, self._catalog, self._counter(), self.settings
            )
            context = {"source_events": sources}
        else:
            raise ValueError("mode must be active_memory, full_history, bm25, or state")
        completion = self._provider.complete(
            instructions=self._reader_instructions,
            content={
                "entity_catalog": deepcopy(self._catalog),
                "context": context,
                "query": query["text"],
                "requested_facts": query["requested_facts"],
            },
            max_output_tokens=self.settings.reader_max_output_tokens,
        )
        keys = [(item["entity_id"], item["field"]) for item in query["requested_facts"]]
        return validate_facts(
            completion.text,
            keys,
            {event["event_id"] for event in sources},
            self.settings,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return portable JSON-serializable state; snapshots contain workflow text."""
        value = {
            "schema_version": self._snapshot_version,
            "baseline_only": self._baseline_only,
            "entity_catalog": self.entity_catalog,
            "settings": asdict(self.settings),
            "events": self.events,
            "state": self.state,
        }
        if self._snapshot_version == 2:
            value["prompts"] = {
                "updater": self._updater_instructions,
                "reader": self._reader_instructions,
            }
        return value

    @classmethod
    def from_events(
        cls,
        entity_catalog: list[dict[str, Any]],
        events: list[dict[str, Any]],
        provider: Any,
        settings: Settings | None = None,
        count_tokens: TokenCounter | None = None,
        *,
        updater_instructions: str = UPDATER_INSTRUCTIONS,
        reader_instructions: str = READER_INSTRUCTIONS,
    ) -> "Threadmark":
        """Load an archive for full-history or BM25 baselines without updater calls.

        Current state remains unknown. Use ``answer(..., mode='full_history')``
        or ``answer(..., mode='bm25')``
        to evaluate the baseline even when a separate active ingestion fails.
        Active answers are disabled until ``ingest([])`` builds the loaded
        archive's state; that operation makes paid updater calls.
        """
        engine = cls(
            entity_catalog,
            provider,
            settings,
            count_tokens,
            updater_instructions=updater_instructions,
            reader_instructions=reader_instructions,
        )
        archive = public_events(events)
        engine._events = archive
        engine._baseline_only = True
        return engine

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, Any] | str,
        *,
        provider: Any,
        count_tokens: TokenCounter | None = None,
    ) -> "Threadmark":
        """Restore a validated snapshot without calling the provider.

        Structural checks do not establish a snapshot's authenticity. Only load
        snapshots from sources trusted to define the workflow's current memory.
        """
        if isinstance(snapshot, str):
            if len(snapshot.encode("utf-8")) > MAX_ARCHIVE_BYTES + 2_000_000:
                raise ValueError("Snapshot exceeds the bounded byte size")
            snapshot = parse_json(snapshot)
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid or unsupported snapshot")
        version = snapshot.get("schema_version")
        expected_keys = {
            "schema_version",
            "baseline_only",
            "entity_catalog",
            "settings",
            "events",
            "state",
        }
        if type(version) is int and version == 2:
            expected_keys.add("prompts")
        if (
            type(version) is not int
            or version not in (1, 2)
            or set(snapshot) != expected_keys
            or type(snapshot["baseline_only"]) is not bool
            or not isinstance(snapshot["settings"], dict)
            or set(snapshot["settings"]) != set(asdict(Settings()))
        ):
            raise ValueError("Invalid or unsupported snapshot")
        prompts = {"updater": UPDATER_INSTRUCTIONS, "reader": READER_INSTRUCTIONS}
        if version == 2:
            supplied = snapshot["prompts"]
            if not isinstance(supplied, dict) or set(supplied) != {"updater", "reader"}:
                raise ValueError("Invalid snapshot prompts")
            prompts = {
                key: _instructions(supplied[key], f"snapshot {key} prompt")
                for key in ("updater", "reader")
            }
        try:
            settings = Settings(**snapshot["settings"])
        except TypeError as exc:
            raise ValueError("Invalid snapshot settings") from exc
        engine = cls(
            snapshot["entity_catalog"],
            provider,
            settings,
            count_tokens,
            updater_instructions=prompts["updater"],
            reader_instructions=prompts["reader"],
        )
        engine._snapshot_version = version
        events = public_events(snapshot["events"])
        # Baseline-only archives can include events too large for an updater
        # chunk, allowing the full-history arm to survive an ingestion failure.
        if not snapshot["baseline_only"]:
            chunks_for(events, engine._counter(), settings)
        state = validate_facts(
            snapshot["state"],
            engine._keys,
            {event["event_id"] for event in events},
            settings,
            memory=True,
        )
        if snapshot["baseline_only"] and state != initial_state(engine._keys):
            raise ValueError(
                "Archive-only baseline cannot contain an ingested memory state"
            )
        engine._events = deepcopy(events)
        engine._state = state
        engine._baseline_only = snapshot["baseline_only"]
        return engine
