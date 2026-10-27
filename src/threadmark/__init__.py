"""Threadmark: workflow memory that selects original evidence for a reader."""

from .config import Settings
from .engine import Threadmark
from .budget import BudgetLedger, Prices
from .providers import Completion, OpenAIProvider, ProviderError
from .local_provider import LocalTransformersProvider
from .retrieval import select_bm25_sources

__all__ = [
    "Settings",
    "Threadmark",
    "BudgetLedger",
    "Prices",
    "Completion",
    "OpenAIProvider",
    "ProviderError",
    "LocalTransformersProvider",
    "select_bm25_sources",
]

__version__ = "0.1.0"
