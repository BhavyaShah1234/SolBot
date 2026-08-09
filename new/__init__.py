"""SolBot v2: an agentic retrieval assistant for ASU Research Computing."""

from .new.config import Config, load_config
from .new.llm import LLM
from .new.memory import Memory
from .new.orchestrator import SolBot
from .new.retrieval import Retriever
from .new.store import build_store
from .new.tools import build_registry
from .new.worker import Worker

__all__ = [
    "Config", "load_config", "LLM", "Memory", "SolBot",
    "Retriever", "build_store", "build_registry", "Worker", "build_app",
]
__version__ = "2.0.0"


def build_app(config_path: str = "config.yaml", on_event=None) -> SolBot:
    """Wire every component together from a single configuration file.

    This is the only assembly point in the codebase. Nothing else constructs
    a store, a retriever or a registry, so swapping a backend means editing
    config.yaml, not hunting through modules.
    """
    from .new.ingest import KnowledgeBase

    cfg = load_config(config_path)
    llm = LLM(cfg)
    store = build_store(cfg)
    knowledge_base = KnowledgeBase(cfg, llm, store)
    retriever = Retriever(cfg, llm, store)
    retriever.refresh_indexes()
    registry = build_registry(cfg, retriever, knowledge_base)
    worker = Worker(cfg, llm, registry)
    memory = Memory(cfg, llm)
    return SolBot(cfg, llm, worker, memory, knowledge_base, on_event)
