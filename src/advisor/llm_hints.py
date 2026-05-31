"""LLM-based PostgreSQL tuning hints via Gemini Flash.

The rule-based advisor covers index type selection and core pgvector
parameters (ef_search, probes, m, lists).  This module uses a lightweight
Gemini Flash model to suggest *additional* PostgreSQL GUC settings that are
hard to encode as simple rules but can meaningfully affect vector-search
performance:

  • random_page_cost / seq_page_cost — planner cost constants (SSD vs HDD)
  • effective_cache_size            — how much OS page-cache to assume
  • jit                             — JIT compilation (overhead vs gain)
  • max_parallel_workers_per_gather — parallelism for IVFFlat / seq scans

The LLM call is best-effort: if the API key is absent or the call fails,
the function returns None and the advisor proceeds with rule-based settings
only.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.profiler.workload_profiler import WorkloadProfile


@dataclass
class LLMHints:
    """Additional PostgreSQL GUC settings suggested by the LLM."""

    session_settings: dict[str, str]
    rationale: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    """Resolve the Gemini API key.

    Priority:
      1. GEMINI_API_KEY environment variable.
      2. api_test/api_key file at the project root (developer convenience).
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # Walk up from this file to find the project root (contains api_test/)
    candidate = Path(__file__).resolve()
    for _ in range(6):          # at most 6 levels up
        candidate = candidate.parent
        key_file = candidate / "api_test" / "api_key"
        if key_file.exists():
            return key_file.read_text().strip()

    return None


def _build_prompt(profile: WorkloadProfile) -> str:
    return f"""\
You are a PostgreSQL performance tuning expert specialising in pgvector workloads.

The rule-based advisor has already decided the vector index type and its core
parameters (ef_search / probes / m / lists / work_mem / maintenance_work_mem).
Your job is to suggest *additional* PostgreSQL session-level GUC settings that
improve query execution for this specific workload.  Focus on settings the
rule-based system does NOT touch:

  - random_page_cost, seq_page_cost
  - effective_cache_size
  - jit (on/off)
  - max_parallel_workers_per_gather
  - enable_seqscan, enable_indexscan (only when genuinely useful to force a plan)

Workload profile:
  n_vectors              : {profile.n_vectors:,}
  dimensions             : {profile.dim}
  top-k                  : {profile.k}
  has_filters            : {profile.has_filters}
  filter_selectivity     : {profile.filter_selectivity:.3%}
  update_rate            : {profile.update_rate:.0%}
  memory_budget_mb       : {profile.memory_budget_mb}
  latency_budget_ms (p95): {profile.latency_budget_ms}
  cache_regime           : {profile.cache_regime}

Rules:
  • Return AT MOST 4 settings.
  • Only include settings that meaningfully differ from PostgreSQL defaults for
    this workload profile.
  • Do NOT include ef_search, probes, m, lists, work_mem, or
    maintenance_work_mem — those are already handled.
  • Respond with ONLY a JSON object — no markdown, no explanation outside JSON.

Required format:
{{
  "settings": {{
    "param_name": "value"
  }},
  "rationale": "one sentence explaining the most important tradeoff"
}}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_llm_tuning_hints(profile: WorkloadProfile) -> LLMHints | None:
    """Query Gemini Flash for additional PostgreSQL GUC tuning suggestions.

    Args:
        profile: WorkloadProfile describing the workload.

    Returns:
        LLMHints with additional session settings, or None if unavailable.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    try:
        from google import genai  # type: ignore[import]
    except ImportError:
        return None

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=_build_prompt(profile),
        )
        raw = response.text.strip()

        # Strip markdown code fences if present
        if "```" in raw:
            parts = raw.split("```")
            # parts[1] is the fenced block; strip optional language tag
            raw = parts[1].lstrip("json").strip()

        data = json.loads(raw)
        return LLMHints(
            session_settings={
                str(k): str(v) for k, v in data.get("settings", {}).items()
            },
            rationale=str(data.get("rationale", "")),
        )

    except Exception:
        # Network errors, quota exceeded, malformed JSON — all silently ignored.
        return None
