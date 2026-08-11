"""The ``agents`` package: SolBot's Phase 2 orchestrator/worker/insight-extractor.

Sits on top of three independent data/tool layers -- ``db_engine``
(vector store), ``memory_engine`` (conversation memory, facts, cross-session
recall), and ``web_search`` (open-web search/fetch) -- adding the
LLM-reasoning layer: query contextualization, decomposition into a DAG of
sub-tasks, worker agents that research each sub-task via tools, verification,
and synthesis into one reply. No LangGraph -- the orchestrator
(:class:`~agents.orchestrator.Session`) is a plain bounded Python loop.

Public surface:

* :class:`~agents.orchestrator.Session` -- one conversation; ``ask(query)``
  runs a full turn.
* :func:`~agents.config.read_config` -- loads ``config.yaml`` with this
  package's section defaults merged in.
"""

from agents.config import read_config
from agents.orchestrator import Session

__all__ = ["Session", "read_config"]
