"""VecAdvisor++ core advisor interface.

Orchestrates workload profiling and rule-based recommendation generation.
"""

from src.advisor.llm_hints import get_llm_tuning_hints
from src.advisor.rules import Recommendation, generate_recommendation
from src.advisor.sql_generator import generate_full_recommendation_sql
from src.profiler.workload_profiler import WorkloadProfile, profile_from_params, profile_from_table


class VecAdvisor:
    """Filter-aware vector index advisor for PostgreSQL with pgvector."""

    def analyze(
        self,
        profile: WorkloadProfile,
        use_llm_hints: bool = True,
    ) -> Recommendation:
        """Analyze a workload profile and produce an index recommendation.

        After the rule-based recommendation is generated, this method
        optionally calls Gemini Flash (via ``get_llm_tuning_hints``) to
        append additional PostgreSQL GUC settings that are workload-specific
        but outside the rule system's scope (e.g. random_page_cost, jit).
        The LLM call is best-effort: if the API key is absent or the call
        fails, the rule-based result is returned unchanged.

        Args:
            profile: WorkloadProfile describing the workload.
            use_llm_hints: If True (default), attempt to enrich the
                recommendation with Gemini Flash GUC suggestions.

        Returns:
            Recommendation with index type, params, and explanations.
        """
        rec = generate_recommendation(profile)

        if use_llm_hints:
            hints = get_llm_tuning_hints(profile)
            if hints:
                # LLM settings fill in GUC knobs not covered by the rules.
                # Rule-based values take precedence if there is a conflict.
                merged = dict(hints.session_settings)
                merged.update(rec.session_settings)  # rules win on collision
                rec.session_settings = merged
                if hints.rationale:
                    rec.explanation.append(f"[LLM] {hints.rationale}")

        return rec

    def analyze_from_db(
        self,
        conn,
        table_name: str,
        vector_column: str = "embedding",
        k: int = 10,
        filter_columns: list[str] | None = None,
        filter_clauses: list[str] | None = None,
        update_rate: float = 0.0,
        memory_budget_mb: int = 2048,
        latency_budget_ms: float = 50.0,
        cache_regime: str = "warm",
    ) -> Recommendation:
        """Analyze by profiling a live PostgreSQL table.

        Args:
            conn: psycopg2 connection.
            table_name: Table to analyze.
            vector_column: Vector column name.
            k: Top-k query parameter.
            filter_columns: Columns used in WHERE clauses.
            filter_clauses: WHERE clauses for selectivity estimation.
            update_rate: Write operation fraction.
            memory_budget_mb: Memory budget for indexes.
            latency_budget_ms: Target p95 latency.
            cache_regime: "cold" or "warm".

        Returns:
            Recommendation.
        """
        profile = profile_from_table(
            conn, table_name, vector_column, k,
            filter_columns, filter_clauses,
            update_rate, memory_budget_mb, latency_budget_ms, cache_regime,
        )
        return self.analyze(profile)

    def analyze_from_params(
        self,
        n_vectors: int,
        dim: int,
        k: int = 10,
        has_filters: bool = False,
        filter_selectivity: float = 1.0,
        filter_columns: list[str] | None = None,
        update_rate: float = 0.0,
        memory_budget_mb: int = 2048,
        latency_budget_ms: float = 50.0,
        cache_regime: str = "warm",
    ) -> Recommendation:
        """Analyze using manually specified parameters (no DB connection).

        Args:
            n_vectors: Dataset size.
            dim: Vector dimensionality.
            k: Top-k.
            has_filters: Whether queries include WHERE clauses.
            filter_selectivity: Estimated filter selectivity.
            filter_columns: Column names used in filters.
            update_rate: Write rate.
            memory_budget_mb: Memory budget.
            latency_budget_ms: Latency target.
            cache_regime: "cold" or "warm".

        Returns:
            Recommendation.
        """
        profile = profile_from_params(
            n_vectors, dim, k, has_filters, filter_selectivity,
            filter_columns, update_rate, memory_budget_mb,
            latency_budget_ms, cache_regime,
        )
        return self.analyze(profile)

    def get_sql(
        self,
        recommendation: Recommendation,
        table_name: str,
        column: str = "embedding",
    ) -> str:
        """Generate executable SQL for a recommendation.

        Args:
            recommendation: The advisor's recommendation.
            table_name: Target table name.
            column: Vector column name.

        Returns:
            Complete SQL string ready to execute.
        """
        return generate_full_recommendation_sql(
            recommendation, table_name, column
        )

    def print_recommendation(
        self,
        recommendation: Recommendation,
        table_name: str,
        column: str = "embedding",
    ) -> None:
        """Print a human-readable recommendation with SQL.

        Args:
            recommendation: The advisor's recommendation.
            table_name: Target table name.
            column: Vector column name.
        """
        print("=" * 70)
        print("VecAdvisor++ Recommendation")
        print("=" * 70)
        print(f"\nIndex Type: {recommendation.index_type.upper()}")

        if recommendation.build_params:
            print(f"Build Parameters: {recommendation.build_params}")
        if recommendation.query_params:
            print(f"Query Parameters: {recommendation.query_params}")

        print("\n--- Reasoning ---")
        for i, exp in enumerate(recommendation.explanation, 1):
            print(f"  {i}. {exp}")

        if recommendation.auxiliary_indexes:
            print("\n--- Auxiliary Indexes ---")
            for aux in recommendation.auxiliary_indexes:
                print(f"  - B-tree on '{aux['column']}': {aux['reason']}")

        print("\n--- Executable SQL ---")
        sql = self.get_sql(recommendation, table_name, column)
        print(sql)
        print("=" * 70)
