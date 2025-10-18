"""
Pipeline Analysis Functions

This module provides functions for analyzing market pipeline results including:
- Loading pipeline run data
- Creating visualizations (capital allocation, volume vs probability, etc.)
- Generating LOB plots

All functions are designed to be imported into Jupyter notebooks for streamlined analysis.
"""

# Standard library
import json
from pathlib import Path
from datetime import datetime

# Third-party
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Local
from utils import data_dir as get_data_dir


def list_pipeline_runs(base_dir=None):
    """
    List all available pipeline runs.

    Args:
        base_dir: Base directory containing pipeline outputs (default: None, uses data_dir() / 'pipelines')

    Returns:
        DataFrame with run information including run_id, date, time, n_questions, etc.
    """
    # Set default base directory if not provided
    if base_dir is None:
        base_dir = str(get_data_dir() / "pipelines")

    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Pipeline directory not found: {base_path}")
        return pd.DataFrame()

    runs = []
    for run_dir in sorted(base_path.iterdir()):
        if not run_dir.is_dir():
            continue

        # Check if it has the expected files
        results_csv = run_dir / "results.csv"
        summary_json = run_dir / "summary.json"
        config_json = run_dir / "pipeline_config.json"

        if not results_csv.exists():
            continue

        run_info = {
            "run_id": run_dir.name,
            "path": str(run_dir),
            "has_results": results_csv.exists(),
            "has_summary": summary_json.exists(),
            "has_config": config_json.exists(),
        }

        # Load summary if available
        if summary_json.exists():
            with open(summary_json) as f:
                summary = json.load(f)
                run_info["n_questions"] = summary.get("n_questions")
                run_info["total_capital"] = summary.get("total_capital")
                run_info["total_allocated"] = summary.get("total_allocated")

        # Parse timestamp
        try:
            timestamp = datetime.strptime(run_dir.name, "%Y%m%d_%H%M")
            run_info["timestamp"] = timestamp
            run_info["date"] = timestamp.strftime("%Y-%m-%d")
            run_info["time"] = timestamp.strftime("%H:%M")
        except ValueError:
            run_info["timestamp"] = None
            run_info["date"] = None
            run_info["time"] = None

        runs.append(run_info)

    if not runs:
        print("No pipeline runs found")
        return pd.DataFrame()

    df = pd.DataFrame(runs)
    df = df.sort_values("run_id", ascending=False).reset_index(drop=True)
    return df


def load_pipeline_run(run_id, base_dir=None):
    """
    Load all data from a pipeline run.

    Args:
        run_id: Run identifier (e.g., "20251016_1126")
        base_dir: Base directory for pipeline outputs (default: None, uses data_dir() / 'pipelines')

    Returns:
        dict with keys: 'results', 'config', 'summary', 'results_json'
        - results: DataFrame with pipeline results
        - config: Dict with pipeline configuration
        - summary: Dict with summary statistics
        - results_json: Full results including nested data
    """
    # Set default base directory if not provided
    if base_dir is None:
        base_dir = str(get_data_dir() / "pipelines")

    run_dir = Path(base_dir) / run_id

    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")

    data = {}

    # Load results CSV
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        data["results"] = pd.read_csv(results_csv)
        print(f"✓ Loaded results: {len(data['results'])} questions")
    else:
        print("✗ results.csv not found")
        data["results"] = None

    # Load results JSON (has full data including probability samples)
    results_json = run_dir / "results.json"
    if results_json.exists():
        with open(results_json) as f:
            data["results_json"] = json.load(f)
        print(f"✓ Loaded full results JSON")
    else:
        print("✗ results.json not found")
        data["results_json"] = None

    # Load config
    config_json = run_dir / "pipeline_config.json"
    if config_json.exists():
        with open(config_json) as f:
            data["config"] = json.load(f)
        print(f"✓ Loaded configuration")
    else:
        print("✗ pipeline_config.json not found")
        data["config"] = None

    # Load summary
    summary_json = run_dir / "summary.json"
    if summary_json.exists():
        with open(summary_json) as f:
            data["summary"] = json.load(f)
        print(f"✓ Loaded summary")
    else:
        print("✗ summary.json not found")
        data["summary"] = None

    return data


def plot_capital_allocation(df, top_n=10, height=600):
    """
    Create a pie chart showing capital allocation across questions.

    Args:
        df: DataFrame with 'allocated_capital' and 'question' columns
        top_n: Number of top allocations to show (default: 10)
        height: Height of the plot in pixels (default: 600)

    Returns:
        plotly Figure object
    """
    if df is None or "allocated_capital" not in df.columns:
        print("Cannot create capital allocation plot - missing required columns")
        return None

    # Sort by allocated capital
    df_sorted = df.sort_values("allocated_capital", ascending=False)

    # Print summary
    print("Top {} Capital Allocations".format(top_n))
    print("=" * 80)
    for i, row in df_sorted.head(top_n).iterrows():
        print(f"${row['allocated_capital']:>10,.2f} - {row['question'][:65]}...")
    print()

    # Create pie chart
    fig = px.pie(
        df_sorted.head(top_n),
        values="allocated_capital",
        names="question",
        title=f"Top {top_n} Capital Allocations",
        height=height,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")

    return fig


def plot_volume_vs_probability(df, height=600, width=1000):
    """
    Create a scatter plot of volume vs probability with capital as bubble size.

    Args:
        df: DataFrame with 'estimated_daily_volume', 'probability_mean',
            'allocated_capital', 'question', and 'topic' columns
        height: Height of the plot in pixels (default: 600)
        width: Width of the plot in pixels (default: 1000)

    Returns:
        plotly Figure object
    """
    if df is None:
        print("Cannot create volume vs probability plot - no data")
        return None

    required_cols = ["estimated_daily_volume", "probability_mean"]
    if not all(col in df.columns for col in required_cols):
        print(
            f"Cannot create volume vs probability plot - missing required columns: {required_cols}"
        )
        return None

    # Filter out missing values
    df_valid = df.dropna(subset=required_cols)

    if len(df_valid) == 0:
        print("No valid data for volume vs probability plot")
        return None

    fig = px.scatter(
        df_valid,
        x="probability_mean",
        y="estimated_daily_volume",
        size="allocated_capital" if "allocated_capital" in df_valid.columns else None,
        hover_data=["question", "topic"],
        title="Volume vs Probability (size = allocated capital)",
        labels={
            "probability_mean": "Probability of YES",
            "estimated_daily_volume": "Estimated Daily Volume ($)",
        },
        height=height,
        width=width,
    )
    fig.update_layout(font=dict(size=14))

    return fig


def plot_lob_grid(
    df,
    Q=1000,
    B=10000,
    half_spread_bps=5,
    max_order_bps=500,
    num_orders_coarse=5,
    num_orders_fine=50,
    max_title_len=60,
):
    """
    Create a grid of LOB plots for all markets in the dataframe.

    Args:
        df: DataFrame with 'probability_mean', 'allocated_capital', and 'question' columns
        Q: Total shares outstanding (default: 1000)
        B: Market depth parameter (default: 10000)
        half_spread_bps: Half spread in basis points (default: 5)
        max_order_bps: Maximum order distance in basis points (default: 500)
        num_orders_coarse: Number of discrete order points (default: 5)
        num_orders_fine: Number of points for fine schedule line (default: 50)
        max_title_len: Maximum length for question titles (default: 60)
    """
    import numpy as np
    from make_market import create_lob_data

    # Filter valid rows
    df_valid = df.dropna(subset=["probability_mean", "allocated_capital"])

    if len(df_valid) == 0:
        print("No valid data for LOB plots")
        return

    n_markets = len(df_valid)

    # Calculate grid dimensions
    n_cols = min(2, n_markets)  # Max 2 columns
    n_rows = (n_markets + n_cols - 1) // n_cols  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

    # Handle single plot case
    if n_markets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each market
    for idx, (_, row) in enumerate(df_valid.iterrows()):
        ax = axes[idx]

        probability = row["probability_mean"]
        capital = row["allocated_capital"]
        question = row["question"]

        # Truncate question if too long
        if len(question) > max_title_len:
            question = question[:max_title_len] + "..."

        # Generate LOB data
        lob_data = create_lob_data(
            probability=probability,
            capital=capital,
            Q=Q,
            B=B,
            half_spread_bps=half_spread_bps,
            max_order_bps=max_order_bps,
            num_orders_coarse=num_orders_coarse,
            num_orders_fine=num_orders_fine,
        )

        # Plot fine schedule as line
        ax.plot(
            lob_data["fine_prices"],
            lob_data["fine_lob"],
            "b-",
            linewidth=2,
            alpha=0.6,
            label="Full LOB",
        )

        # Plot coarse schedule as dots
        ax.scatter(
            lob_data["coarse_prices"],
            lob_data["coarse_lob"],
            c="red",
            s=50,
            zorder=5,
            label="Discrete orders",
        )

        # Formatting
        ax.set_title(question, fontsize=10, wrap=True)
        ax.set_xlabel("Probability", fontsize=10)
        ax.set_ylabel("Cumulative Capital ($)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Add capital annotation
        ax.text(
            0.02,
            0.98,
            f"Capital: ${capital:,.0f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Hide unused subplots
    for idx in range(n_markets, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

    print(f"\nPlotted {n_markets} markets")
    print(f"Grid size: {n_rows} rows × {n_cols} columns")


def format_results_display(df):
    """
    Format results DataFrame for display with nice formatting.

    Args:
        df: DataFrame with pipeline results

    Returns:
        Formatted DataFrame ready for display
    """
    if df is None:
        return None

    # Select columns to display
    display_cols = [
        "topic",
        "question",
        "estimated_daily_volume",
        "probability_mean",
        "probability_std",
        "allocated_capital",
    ]

    # Filter to available columns
    display_cols = [col for col in display_cols if col in df.columns]

    df_display = df[display_cols].copy()

    # Format numeric columns
    if "estimated_daily_volume" in df_display.columns:
        df_display["estimated_daily_volume"] = df_display[
            "estimated_daily_volume"
        ].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")

    if "allocated_capital" in df_display.columns:
        df_display["allocated_capital"] = df_display["allocated_capital"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
        )

    if "probability_mean" in df_display.columns:
        df_display["probability_mean"] = df_display["probability_mean"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )

    if "probability_std" in df_display.columns:
        df_display["probability_std"] = df_display["probability_std"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )

    return df_display
