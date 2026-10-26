"""Configuration for the evaluated memory-guided evidence method."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Settings:
    """Model and context limits; defaults match the published workflow study.

    Changing these values creates a new configuration whose accuracy must be
    evaluated separately. Configure the provider with the same model and effort.
    """

    model: str = "gpt-5.6-luna"
    reasoning_effort: str = "none"
    chunk_tokens: int = 2000
    updater_max_output_tokens: int = 1024
    reader_max_output_tokens: int = 256
    max_memory_bytes: int = 12000
    recent_source_tokens: int = 1024
    max_source_bytes: int = 20000
    max_value_chars: int = 256
    max_evidence_ids: int = 2

    def __post_init__(self) -> None:
        for name in ("model", "reasoning_effort"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise ValueError(f"{name} must be a nonempty bounded string")
        for field in fields(self):
            if field.name in {"model", "reasoning_effort"}:
                continue
            value = getattr(self, field.name)
            if type(value) is not int or not 1 <= value <= 1_000_000:
                raise ValueError(
                    f"{field.name} must be an integer between 1 and 1000000"
                )
