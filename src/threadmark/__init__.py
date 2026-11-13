"""Threadmark: workflow memory that selects original evidence for a reader."""

from .config import Settings
from .engine import Threadmark
from .budget import BudgetLedger, Prices
from .providers import Completion, OpenAIProvider, ProviderError
from .local_provider import LocalTransformersProvider
from .retrieval import select_bm25_sources
from .active_context import (
    ActiveContext,
    CompiledContext,
    ContextBudgetExceeded,
    ACTIVE_CONTEXT_INSTRUCTIONS,
    render_json_request,
)
from .workflow_memory import (
    WorkflowMemory,
    WORKFLOW_UPDATER_INSTRUCTIONS,
    WORKFLOW_HISTORY_UPDATER_INSTRUCTIONS,
    WORKFLOW_READER_INSTRUCTIONS,
)

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
    "ActiveContext",
    "CompiledContext",
    "ContextBudgetExceeded",
    "ACTIVE_CONTEXT_INSTRUCTIONS",
    "render_json_request",
    "WorkflowMemory",
    "WORKFLOW_UPDATER_INSTRUCTIONS",
    "WORKFLOW_HISTORY_UPDATER_INSTRUCTIONS",
    "WORKFLOW_READER_INSTRUCTIONS",
]

__version__ = "0.2.0"
