"""
Market Calibration Plot

This script analyzes how well-calibrated Polymarket predictions are by comparing
market prices to actual resolutions.

For each price bin, we compute:
- The fraction of markets that resolved to 1
- Confidence intervals using binomial statistics

A well-calibrated market should show points along the diagonal (45-degree line).
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from polymarket_data import load_all_market_histories


def compute_binomial_ci(successes, trials, confidence=0.95):
    """
    Compute confidence interval for binomial proportion using Wilson score interval.

    Args:
        successes: Number of successes
        trials: Number of trials
        confidence: Confidence level (default: 0.95)

    Returns:
        (lower_bound, upper_bound)
    """
    if trials == 0:
        return (0, 0)

    # Wilson score interval (better for small samples and edge cases)
    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator

    return (max(0, center - margin), min(1, center + margin))


def create_calibration_data(df, n_bins=100, cut_by=None, cut_bins=4):
    """
    Create calibration data by binning prices and computing resolution rates.

    Optionally stratify by an additional field (e.g., volume) to compare calibration
    across different market characteristics.

    Args:
        df: DataFrame from load_all_market_histories()
        n_bins: Number of price bins (default: 100)
        cut_by: Optional column name to stratify by (e.g., 'avg_daily_volume').
                If None, returns non-stratified calibration data. (default: None)
        cut_bins: Number of quantiles to create for stratification (default: 4)

    Returns:
        DataFrame with columns:
            - bin_center: Center of price bin
            - bin_min: Minimum price in bin
            - bin_max: Maximum price in bin
            - resolution_rate: Fraction resolving to 1
            - count: Number of observations in bin
            - ci_lower: Lower confidence bound
            - ci_upper: Upper confidence bound

        If cut_by is specified, also includes:
            - {cut_by}_quantile: Quantile label (e.g., 'Q1', 'Q2', etc.)
            - {cut_by}_range: Range of values in this quantile
    """
    df = df.copy()

    # If stratification requested, create quantiles
    quantile_ranges = None
    if cut_by is not None:
        df[f"{cut_by}_quantile"], quantile_bins = pd.qcut(
            df[cut_by],
            q=cut_bins,
            labels=[f"Q{i+1}" for i in range(cut_bins)],
            retbins=True,
            duplicates="drop",
        )

        # Store quantile ranges for labeling
        quantile_ranges = {}
        for i, (low, high) in enumerate(zip(quantile_bins[:-1], quantile_bins[1:])):
            quantile_ranges[f"Q{i+1}"] = (low, high)

    # Create price bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Assign each price to a bin
    df["price_bin"] = pd.cut(
        df["price"], bins=bins, labels=bin_centers, include_lowest=True
    )

    # Group by bin and compute statistics
    results = []

    # Determine grouping levels
    if cut_by is not None:
        # Stratified case: iterate over quantiles
        quantile_col = f"{cut_by}_quantile"
        for quantile in df[quantile_col].dropna().unique():
            quantile_data = df[df[quantile_col] == quantile]

            for bin_center in bin_centers:
                bin_data = quantile_data[quantile_data["price_bin"] == bin_center]

                if len(bin_data) == 0:
                    continue

                # Count how many resolved to 1
                n_total = len(bin_data)
                n_resolved_to_1 = (bin_data["final_price"] == 1.0).sum()
                resolution_rate = n_resolved_to_1 / n_total

                # Compute confidence interval
                ci_lower, ci_upper = compute_binomial_ci(n_resolved_to_1, n_total)

                # Find actual bin range
                bin_min = bin_data["price"].min()
                bin_max = bin_data["price"].max()

                # Get quantile range
                q_low, q_high = quantile_ranges[quantile]

                results.append(
                    {
                        "bin_center": bin_center,
                        "bin_min": bin_min,
                        "bin_max": bin_max,
                        "resolution_rate": resolution_rate,
                        "count": n_total,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        f"{cut_by}_quantile": quantile,
                        f"{cut_by}_range": f"[{q_low:.0f}, {q_high:.0f}]",
                    }
                )
    else:
        # Non-stratified case: single calibration curve
        for bin_center in bin_centers:
            bin_data = df[df["price_bin"] == bin_center]

            if len(bin_data) == 0:
                continue

            # Count how many resolved to 1
            n_total = len(bin_data)
            n_resolved_to_1 = (bin_data["final_price"] == 1.0).sum()
            resolution_rate = n_resolved_to_1 / n_total

            # Compute confidence interval
            ci_lower, ci_upper = compute_binomial_ci(n_resolved_to_1, n_total)

            # Find actual bin range
            bin_min = bin_data["price"].min()
            bin_max = bin_data["price"].max()

            results.append(
                {
                    "bin_center": bin_center,
                    "bin_min": bin_min,
                    "bin_max": bin_max,
                    "resolution_rate": resolution_rate,
                    "count": n_total,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

    return pd.DataFrame(results)


def plot_calibration(calibration_df, title="Polymarket Calibration Plot"):
    """
    Create an interactive calibration plot with confidence intervals.

    Args:
        calibration_df: DataFrame from create_calibration_data()
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Add confidence interval as shaded area
    fig.add_trace(
        go.Scatter(
            x=np.concatenate(
                [calibration_df["bin_center"], calibration_df["bin_center"][::-1]]
            ),
            y=np.concatenate(
                [calibration_df["ci_upper"], calibration_df["ci_lower"][::-1]]
            ),
            fill="toself",
            fillcolor="rgba(0, 100, 200, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="95% Confidence Interval",
            hoverinfo="skip",
        )
    )

    # Add main calibration line
    fig.add_trace(
        go.Scatter(
            x=calibration_df["bin_center"],
            y=calibration_df["resolution_rate"],
            mode="markers+lines",
            name="Observed Resolution Rate",
            line=dict(color="rgb(0, 100, 200)", width=2),
            marker=dict(
                size=calibration_df["count"] / calibration_df["count"].max() * 20 + 5,
                color=calibration_df["count"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Count", x=1.15),
                line=dict(width=1, color="white"),
            ),
            text=calibration_df.apply(
                lambda row: f"Price: {row['bin_center']:.3f}<br>"
                + f"Resolution Rate: {row['resolution_rate']:.3f}<br>"
                + f"Count: {row['count']}<br>"
                + f"CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]",
                axis=1,
            ),
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Add perfect calibration line (diagonal)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="red", width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        xaxis_title=dict(text="Market Price", font=dict(size=18)),
        yaxis_title=dict(text="Fraction Resolving to 1", font=dict(size=18)),
        hovermode="closest",
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            font=dict(size=14), x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        font=dict(size=14),
        plot_bgcolor="white",
        xaxis=dict(range=[0, 1], gridcolor="lightgray", showgrid=True),
        yaxis=dict(range=[0, 1], gridcolor="lightgray", showgrid=True),
    )

    return fig


def plot_stratified_calibration(
    stratified_df, cut_by, title="Stratified Calibration Plot"
):
    """
    Create calibration plot with separate lines for each quantile.

    Args:
        stratified_df: DataFrame from create_calibration_data() with cut_by parameter
        cut_by: Column name used for stratification
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    quantile_col = f"{cut_by}_quantile"
    range_col = f"{cut_by}_range"

    # Define colors for quantiles
    colors = [
        "rgb(31, 119, 180)",
        "rgb(255, 127, 14)",
        "rgb(44, 160, 44)",
        "rgb(214, 39, 40)",
        "rgb(148, 103, 189)",
        "rgb(140, 86, 75)",
        "rgb(227, 119, 194)",
        "rgb(127, 127, 127)",
    ]

    quantiles = sorted(stratified_df[quantile_col].unique())

    # Plot each quantile as a separate line
    for i, quantile in enumerate(quantiles):
        quantile_data = stratified_df[
            stratified_df[quantile_col] == quantile
        ].sort_values("bin_center")

        if len(quantile_data) == 0:
            continue

        # Get the range for this quantile
        q_range = quantile_data[range_col].iloc[0]

        # Add line for this quantile
        fig.add_trace(
            go.Scatter(
                x=quantile_data["bin_center"],
                y=quantile_data["resolution_rate"],
                mode="lines+markers",
                name=f"{quantile}: {q_range}",
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6, line=dict(width=1, color="white")),
                text=quantile_data.apply(
                    lambda row: f"Price: {row['bin_center']:.3f}<br>"
                    + f"Resolution Rate: {row['resolution_rate']:.3f}<br>"
                    + f"Count: {row['count']}<br>"
                    + f"{quantile}: {row[range_col]}",
                    axis=1,
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Add perfect calibration line (diagonal)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="black", width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        xaxis_title=dict(text="Market Price", font=dict(size=18)),
        yaxis_title=dict(text="Fraction Resolving to 1", font=dict(size=18)),
        hovermode="closest",
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            font=dict(size=12), x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        font=dict(size=14),
        plot_bgcolor="white",
        xaxis=dict(range=[0, 1], gridcolor="lightgray", showgrid=True),
        yaxis=dict(range=[0, 1], gridcolor="lightgray", showgrid=True),
    )

    return fig


def print_calibration_statistics(calibration_df):
    """Print summary statistics about calibration."""
    print("\n=== Calibration Statistics ===\n")

    # Overall statistics
    total_obs = calibration_df["count"].sum()
    print(f"Total observations: {total_obs:,}")
    print(f"Number of price bins: {len(calibration_df)}")
    print(
        f"Price range: [{calibration_df['bin_min'].min():.3f}, {calibration_df['bin_max'].max():.3f}]"
    )

    # Calibration error
    calibration_df["abs_error"] = np.abs(
        calibration_df["bin_center"] - calibration_df["resolution_rate"]
    )
    weighted_error = (
        calibration_df["abs_error"] * calibration_df["count"]
    ).sum() / total_obs
    print(f"\nMean Absolute Calibration Error: {weighted_error:.4f}")

    # Bins with most data
    print("\nTop 10 bins by count:")
    top_bins = calibration_df.nlargest(10, "count")[
        ["bin_center", "resolution_rate", "count"]
    ]
    print(top_bins.to_string(index=False))

    # Bins with largest calibration error
    print("\nTop 10 bins by calibration error:")
    error_bins = calibration_df.nlargest(10, "abs_error")[
        ["bin_center", "resolution_rate", "abs_error", "count"]
    ]
    print(error_bins.to_string(index=False))


if __name__ == "__main__":
    print("Loading market histories...")
    df = load_all_market_histories(sample_per_day=True)

    print(f"\nLoaded {len(df):,} observations")
    print(f"Markets: {df['slug'].nunique()}")
    print(f"Outcomes: {df['outcome'].nunique()}")

    # Create calibration data
    print("\nCreating calibration bins...")
    n_bins = 100
    calibration_df = create_calibration_data(df, n_bins=n_bins)

    # Print statistics
    print_calibration_statistics(calibration_df)

    # Create plot
    print("\nCreating calibration plot...")
    fig = plot_calibration(
        calibration_df, title=f"Polymarket Calibration Plot ({n_bins} bins)"
    )

    # Show plot
    fig.show()

    # Optionally save to HTML
    output_file = "polymarket_calibration_plot.html"
    fig.write_html(output_file)
    print(f"\nPlot saved to: {output_file}")
