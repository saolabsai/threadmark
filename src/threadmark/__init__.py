"""Threadmark: workflow memory that selects original evidence for a reader."""

from .config import Settings
from .engine import Threadmark
from .budget import BudgetLedger, Prices
from .providers import Completion, OpenAIProvider, ProviderError
from .local_provider import LocalTransformersProvider
from .retrieval import select_bm25_sources
from .workflow_memory import WorkflowMemory, WORKFLOW_READER_INSTRUCTIONS

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
    "WorkflowMemory",
    "WORKFLOW_READER_INSTRUCTIONS",
]

__version__ = "0.1.0"
