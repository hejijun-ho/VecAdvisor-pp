"""Visualization utilities for VecAdvisor++ evaluation results.

Generates publication-quality plots comparing baseline vs advisor configurations.
All functions save to a specified output directory and return the saved path.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.benchmark.metrics import BenchmarkResult

# ---------------------------------------------------------------------------
# Global publication-quality style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color / marker scheme per config name (consistent across all plots)
_CONFIG_STYLE: dict[str, dict] = {
    "pgvector_default":       {"color": "#1f77b4", "marker": "o", "linestyle": "--"},
    "heuristic_hnsw":         {"color": "#ff7f0e", "marker": "^", "linestyle": "--"},
    "hnsw_aggressive":        {"color": "#9467bd", "marker": "v", "linestyle": "--"},
    "pgvector_ivfflat_default": {"color": "#2ca02c", "marker": "D", "linestyle": "-."},
    "ivfflat_full_probes":    {"color": "#8c564b", "marker": "P", "linestyle": "-."},
    "sequential_scan":        {"color": "#7f7f7f", "marker": "x", "linestyle": ":"},
    "vecadvisor++":           {"color": "#d62728", "marker": "s", "linestyle": "-"},
}

def _style(name: str) -> dict:
    return _CONFIG_STYLE.get(name, {"color": "#17becf", "marker": "o", "linestyle": "-"})


# ---------------------------------------------------------------------------
# 1. Recall vs Latency (with optional error bars)
# ---------------------------------------------------------------------------

def plot_recall_vs_latency(
    results: list[BenchmarkResult],
    output_dir: str = "results/plots",
    filename: str = "recall_vs_latency.png",
) -> str:
    """Scatter plot of recall@k vs p95 latency for all configurations.

    When BenchmarkResult.num_runs > 1, horizontal (latency) and vertical
    (recall) error bars are drawn using the precomputed std fields.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for r in results:
        s = _style(r.config_name)
        size = 120 if r.config_name == "vecadvisor++" else 70
        xerr = r.latency_p95_std if r.num_runs > 1 else None
        yerr = r.recall_std if r.num_runs > 1 else None
        ax.errorbar(
            r.latency_p95_ms, r.recall,
            xerr=xerr, yerr=yerr,
            fmt=s["marker"], color=s["color"],
            markersize=size ** 0.5, capsize=3, capthick=1,
            label=r.config_name, zorder=5,
        )

    ax.set_xlabel("p95 Latency (ms)")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall vs Latency Tradeoff")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Build Time Comparison
# ---------------------------------------------------------------------------

def plot_build_time_comparison(
    results: list[BenchmarkResult],
    output_dir: str = "results/plots",
    filename: str = "build_time.png",
) -> str:
    """Bar chart of index build times."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    names = [r.config_name for r in results]
    build_times = [r.build_time_s for r in results]
    colors = [_style(n)["color"] for n in names]

    bars = ax.bar(names, build_times, color=colors)
    ax.set_ylabel("Build Time (seconds)")
    ax.set_title("Index Build Time Comparison")
    ax.tick_params(axis="x", rotation=30)

    for bar, val in zip(bars, build_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=8,
        )

    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Completion Rate
# ---------------------------------------------------------------------------

def plot_completion_rate(
    results: list[BenchmarkResult],
    output_dir: str = "results/plots",
    filename: str = "completion_rate.png",
) -> str:
    """Bar chart of top-k completion rates with optional error bars."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    names = [r.config_name for r in results]
    rates = [r.completion_rate * 100 for r in results]
    colors = [_style(n)["color"] for n in names]
    errs = [r.completion_rate_std * 100 if r.num_runs > 1 else 0 for r in results]

    bars = ax.bar(names, rates, color=colors, yerr=errs, capsize=4)
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Top-k Completion Rate (Filtered Queries)")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=30)

    for bar, val in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
        )

    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. Selectivity Heatmap
# ---------------------------------------------------------------------------

def plot_selectivity_heatmap(
    results_by_selectivity: dict[str, list[BenchmarkResult]],
    metric: str = "recall",
    output_dir: str = "results/plots",
    filename: str = "selectivity_heatmap.png",
) -> str:
    """Heatmap of a metric across selectivity levels and configurations."""
    os.makedirs(output_dir, exist_ok=True)
    selectivities = list(results_by_selectivity.keys())
    if not selectivities:
        return ""

    config_names = [r.config_name for r in results_by_selectivity[selectivities[0]]]
    data = np.zeros((len(config_names), len(selectivities)))
    for j, sel in enumerate(selectivities):
        for i, r in enumerate(results_by_selectivity[sel]):
            data[i, j] = getattr(r, metric)

    fig, ax = plt.subplots(figsize=(max(8, len(selectivities) * 2), max(4, len(config_names))))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(selectivities)))
    ax.set_xticklabels(selectivities)
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels(config_names)
    ax.set_xlabel("Filter Selectivity")
    ax.set_title(f"{metric} across Selectivity Levels")

    for i in range(len(config_names)):
        for j in range(len(selectivities)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. Scalability Plot  (n vs. metric, one line per config)
# ---------------------------------------------------------------------------

def plot_scalability(
    results_by_n: dict[int, list[BenchmarkResult]],
    metric: str = "recall",
    output_dir: str = "results/plots",
    filename: str = "scalability.png",
) -> str:
    """Line plot of a metric vs. dataset size n for each configuration.

    Args:
        results_by_n: Dict mapping n_vectors (int) → list of BenchmarkResults.
        metric: BenchmarkResult attribute to plot on the y-axis.
        output_dir: Directory to save the plot.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not results_by_n:
        return ""

    ns = sorted(results_by_n.keys())
    config_names = [r.config_name for r in results_by_n[ns[0]]]

    fig, ax = plt.subplots(figsize=(9, 5))

    for cfg in config_names:
        ys, errs = [], []
        for n in ns:
            r = next((x for x in results_by_n[n] if x.config_name == cfg), None)
            if r is None:
                ys.append(float("nan"))
                errs.append(0.0)
            else:
                ys.append(getattr(r, metric))
                std_attr = metric + "_std"
                errs.append(getattr(r, std_attr, 0.0) if r.num_runs > 1 else 0.0)

        s = _style(cfg)
        lw = 2.0 if cfg == "vecadvisor++" else 1.2
        ax.errorbar(
            ns, ys, yerr=errs,
            label=cfg, color=s["color"], marker=s["marker"],
            linestyle=s["linestyle"], linewidth=lw,
            capsize=3, capthick=1,
        )

    ax.set_xlabel("Dataset Size (n vectors)")
    ax.set_xscale("log")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} vs Dataset Size")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. Parameter Sensitivity / Pareto Sweep
# ---------------------------------------------------------------------------

def plot_pareto_sweep(
    sweep_results: list[tuple[float, BenchmarkResult]],
    param_name: str,
    advisor_value: float | None = None,
    output_dir: str = "results/plots",
    filename: str = "pareto_sweep.png",
) -> str:
    """Recall vs p95 latency Pareto curve parameterised by a single knob value.

    Each point on the curve corresponds to one knob value. Points are colored
    by the parameter value (low → blue, high → red). The advisor's chosen value
    is marked with a star if provided.

    Args:
        sweep_results: List of (param_value, BenchmarkResult) pairs.
        param_name: Name of the swept parameter (e.g. "ef_search", "probes").
        advisor_value: Advisor's chosen value — marked with a gold star.
        output_dir: Directory to save the plot.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not sweep_results:
        return ""

    param_vals = np.array([v for v, _ in sweep_results])
    latencies = np.array([r.latency_p95_ms for _, r in sweep_results])
    recalls = np.array([r.recall for _, r in sweep_results])

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        latencies, recalls,
        c=param_vals, cmap="coolwarm", s=80, zorder=5, edgecolors="k", linewidths=0.5,
    )
    fig.colorbar(sc, ax=ax, label=param_name)

    # Connect points in order of param value
    order = np.argsort(param_vals)
    ax.plot(latencies[order], recalls[order], "k--", linewidth=0.8, alpha=0.4, zorder=4)

    # Label each point with the parameter value
    for pv, lat, rec in zip(param_vals, latencies, recalls):
        ax.annotate(f"{pv:.0f}", (lat, rec), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, alpha=0.8)

    # Mark advisor's chosen value
    if advisor_value is not None:
        for pv, lat, rec in zip(param_vals, latencies, recalls):
            if abs(pv - advisor_value) < 1e-6:
                ax.scatter([lat], [rec], marker="*", color="gold", s=300,
                           zorder=6, edgecolors="k", linewidths=0.8,
                           label=f"Advisor ({param_name}={advisor_value:.0f})")
                break

    ax.set_xlabel("p95 Latency (ms)")
    ax.set_ylabel("Recall@k")
    ax.set_title(f"Recall–Latency Tradeoff vs {param_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7. Cross-Dataset Bars (SIFT vs GIST side-by-side)
# ---------------------------------------------------------------------------

def plot_cross_dataset_bars(
    sift_results: list[BenchmarkResult],
    gist_results: list[BenchmarkResult],
    metric: str = "recall",
    output_dir: str = "results/plots",
    filename: str = "cross_dataset.png",
) -> str:
    """Side-by-side grouped bar chart comparing SIFT1M and GIST1M results.

    Args:
        sift_results: BenchmarkResults on SIFT1M.
        gist_results: BenchmarkResults on GIST1M (same config order).
        metric: BenchmarkResult attribute to plot.
        output_dir: Directory to save the plot.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not sift_results or not gist_results:
        return ""

    configs = [r.config_name for r in sift_results]
    sift_vals = [getattr(r, metric) for r in sift_results]
    gist_vals_map = {r.config_name: getattr(r, metric) for r in gist_results}
    gist_vals = [gist_vals_map.get(c, float("nan")) for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, sift_vals, width, label="SIFT1M (128-dim)",
                   color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x + width / 2, gist_vals, width, label="GIST1M (960-dim)",
                   color="#ff7f0e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} — SIFT1M vs GIST1M")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 8. Generate all standard plots
# ---------------------------------------------------------------------------

def generate_all_plots(
    results: list[BenchmarkResult],
    output_dir: str = "results/plots",
) -> list[str]:
    """Generate all standard comparison plots.

    Args:
        results: List of BenchmarkResults.
        output_dir: Output directory.

    Returns:
        List of paths to generated plots.
    """
    paths = [
        plot_recall_vs_latency(results, output_dir),
        plot_build_time_comparison(results, output_dir),
        plot_completion_rate(results, output_dir),
    ]
    return [p for p in paths if p]
