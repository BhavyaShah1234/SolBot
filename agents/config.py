"""Configuration loading for the ``agents`` package.

Unlike ``web_search.config.read_config`` (which returns only its own
``web`` section), this returns the *full* parsed ``config.yaml`` dict —
``agents`` genuinely needs cross-package sections (``retrieval``,
``embedding``, ``vector_store`` for ``db_engine``; ``memory`` for
``memory_engine``) alongside its own (``retrieval``'s fusion keys,
``generation``, ``insight``, ``planner``, ``worker``, ``verification``,
``app``), and every downstream call in this package threads the same
``cfg`` dict through rather than re-loading it per module.

Sections this package owns, or reads keys from, are merged over
:data:`_DEFAULTS` so a missing key degrades to a sane default instead of a
``KeyError`` — the same class of bug ``db_engine.engine.DBEngine.fetch``
originally risked before ``retrieval:`` existed in ``config.yaml`` at all.
This includes ``web`` (mirroring ``web_search/config.py``'s own defaults,
since ``agents/tools.py`` reads ``cfg["web"]`` directly) even though
``agents`` doesn't own that section. Sections this package never reads any
key from (``memory``, ``embedding``, ``vector_store``, ``confluence``,
``chunking``, ``logging``) are passed through untouched.
"""

import yaml

_DEFAULTS = {
    "retrieval": {
        "dense_k": 12,
        "sparse_k": 12,
        "rrf_k_constant": 60,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "use_mmr": True,
        "mmr_lambda": 0.6,
        "final_k": 5,
        "min_evidence_score": 0.55,
        "min_evidence_count": 2,
    },
    "generation": {
        "temperature": 0.1,
        "json_temperature": 0.0,
        "num_ctx": 8192,
        "request_timeout_seconds": 180,
    },
    "insight": {
        "evidence_distill_enabled": True,
        "evidence_distill_max_facts": 6,
        "compression_target_words": 120,
    },
    "planner": {
        "max_nodes": 6,
        "max_depth": 4,
        "decomposition_threshold_words": 8,
        "allow_replanning": True,
        "max_replans": 1,
    },
    "worker": {
        "max_react_steps": 4,
        "answer_max_words": 180,
        "max_parallel_nodes": 3,
        "escalate_to_web_below_confidence": 0.55,
    },
    "verification": {
        "min_overall_confidence": 0.45,
        "escalation_message": "I don't have enough grounded information to answer that confidently.",
        "escalation_contact": "the ASU Research Computing team",
    },
    "app": {
        "timezone": "America/Phoenix",
        "log_file": "agents.log",
        "log_level": "INFO",
    },
    # Mirrors web_search/config.py's own _DEFAULTS exactly -- agents/tools.py reads
    # cfg["web"] directly (make_web_search/make_fetch_url), so this section needs the
    # same KeyError-proofing as every other section here, even though agents/ doesn't
    # own web_search's config.
    "web": {
        "enabled": True,
        "search_provider": "ddgs",
        "extraction_backend": "trafilatura",
        "max_results": 5,
        "search_timeout_seconds": 10,
        "fetch_timeout_seconds": 15,
        "max_fetch_chars": 12000,
        "min_extracted_chars": 200,
        "blocked_domains": [],
    },
}


def read_config(path: str = "config.yaml") -> dict:
    """Loads ``config.yaml`` and merges this package's section defaults into it.

    Args:
        path: Filesystem path to the YAML config file, relative to the
            current working directory unless absolute.

    Returns:
        The full parsed config dict. For each key in :data:`_DEFAULTS`,
        that section is shallow-merged (defaults first, file overrides
        second) so every key this package reads is always present. Every
        other top-level section (``confluence``, ``chunking``,
        ``embedding``, ``vector_store``, ``logging``, ``memory``, ``web``,
        and ``retrieval``/``generation``'s own base keys) is passed
        through exactly as parsed.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        yaml.YAMLError: If the file exists but is not valid YAML.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    for section, defaults in _DEFAULTS.items():
        cfg[section] = {**defaults, **cfg.get(section, {})}
    return cfg
