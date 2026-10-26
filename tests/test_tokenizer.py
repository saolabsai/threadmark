"""Tokenizer downloads require a separate explicit setup operation."""

import hashlib
import io
from types import SimpleNamespace

import pytest
import tiktoken

from threadmark import context

URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
ASSET = b"test-asset"
DIGEST = hashlib.sha256(ASSET).hexdigest()


@pytest.fixture
def configured_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(context, "_encoding_metadata", lambda: (URL, DIGEST))
    return context._cache_path(URL)


@pytest.mark.parametrize("contents", [None, b"corrupt"])
def test_missing_or_corrupt_cache_fails_before_tiktoken_can_fetch(
    configured_cache, monkeypatch, contents
):
    if contents is not None:
        configured_cache.write_bytes(contents)

    def unexpected(*args, **kwargs):
        raise AssertionError("No tokenizer fetch or load is permitted")

    monkeypatch.setattr(tiktoken, "get_encoding", unexpected)
    monkeypatch.setattr(context, "urlopen", unexpected)
    with pytest.raises(ValueError, match="setup-tokenizer"):
        context.token_counter()


def test_verified_cache_loads_evaluated_encoding(configured_cache, monkeypatch):
    configured_cache.write_bytes(ASSET)
    names = []

    def fake_encoding(name):
        names.append(name)
        return SimpleNamespace(encode=lambda text, disallowed_special: [1] * len(text))

    monkeypatch.setattr(tiktoken, "get_encoding", fake_encoding)
    assert context.token_counter()("abc") == 3
    assert names == ["o200k_base"]


def test_setup_is_explicit_verified_and_reuses_valid_cache(
    configured_cache, monkeypatch
):
    requested = []

    def fake_download(url, timeout):
        requested.append((url, timeout))
        return io.BytesIO(ASSET)

    monkeypatch.setattr(context, "urlopen", fake_download)
    first = context.setup_tokenizer()
    second = context.setup_tokenizer()
    assert first["status"] == "downloaded"
    assert second["status"] == "ready"
    assert configured_cache.read_bytes() == ASSET
    assert requested == [(URL, 30)]


def test_setup_rejects_corrupt_download_without_replacing_cache(
    configured_cache, monkeypatch
):
    configured_cache.write_bytes(b"old-corrupt-cache")
    monkeypatch.setattr(
        context, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong-download")
    )
    with pytest.raises(ValueError, match="verification"):
        context.setup_tokenizer()
    assert configured_cache.read_bytes() == b"old-corrupt-cache"


def test_cache_directory_matches_tiktoken_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(tmp_path / "data-gym"))
    assert context._cache_path(URL).parent == tmp_path / "data-gym"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "tiktoken"))
    assert context._cache_path(URL).parent == tmp_path / "tiktoken"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
    with pytest.raises(ValueError, match="disabled"):
        context.token_counter()


def test_official_metadata_has_verified_digest():
    url, digest = context._encoding_metadata()
    assert url == URL
    assert len(digest) == 64
