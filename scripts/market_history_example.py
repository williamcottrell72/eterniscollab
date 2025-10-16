"""
Example usage of market history downloader.

This script demonstrates how to:
1. Download a single market
2. Load downloaded data
3. Perform basic analysis
4. Visualize results
"""

from market_history_downloader import (
    download_market_complete_history,
    load_market_data,
    RateLimiter,
)
import pandas as pd


def download_example():
    """Download a single market example."""
    print("=" * 80)
    print("EXAMPLE 1: Download Single Market")
    print("=" * 80)

    result = download_market_complete_history(
        market_slug="will-donald-trump-be-inaugurated",
        fidelity=10,  # 10-minute bars
        overwrite=False,  # Use cache if available
    )

    print(f"\nDownload {'successful' if result['success'] else 'failed'}!")
    print(f"  Outcomes downloaded: {result.get('outcomes_downloaded', 0)}")
    print(f"  Total data points: {result.get('total_data_points', 0):,}")


def load_and_analyze():
    """Load and analyze downloaded data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Load and Analyze Data")
    print("=" * 80)

    # Load data
    data = load_market_data("will-donald-trump-be-inaugurated")

    print(f"\nMarket: {data['metadata']['question']}")
    print(f"Outcomes: {data['outcome_names']}")
    print(
        f"Date range: {data['metadata']['start_date'][:10]} to {data['metadata']['end_date'][:10]}"
    )

    # Analyze each outcome
    for name, df in zip(data["outcome_names"], data["outcomes"]):
        print(f"\n{name} Outcome:")
        print(f"  Data points: {len(df):,}")
        print(f"  Start price: {df.iloc[0]['price']:.3f}")
        print(f"  End price: {df.iloc[-1]['price']:.3f}")
        print(f"  Price range: {df['price'].min():.3f} - {df['price'].max():.3f}")
        print(f"  Mean price: {df['price'].mean():.3f}")

        # Calculate daily volatility
        df = df.copy()
        df["returns"] = df["price"].pct_change()
        # 10-minute bars: 6 per hour * 24 hours = 144 per day
        daily_vol = df["returns"].std() * (144**0.5)
        print(f"  Daily volatility: {daily_vol:.4f}")


def compare_outcomes():
    """Compare different outcomes."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Compare Outcomes")
    print("=" * 80)

    data = load_market_data("will-donald-trump-be-inaugurated")

    # Get Yes and No outcomes
    yes_df = data["outcomes"][0]
    no_df = data["outcomes"][1]

    # Should sum to ~1 at all times
    print("\nOutcome prices should sum to ~1 (arbitrage-free):")

    # Sample at different times
    for i in [
        0,
        len(yes_df) // 4,
        len(yes_df) // 2,
        3 * len(yes_df) // 4,
        len(yes_df) - 1,
    ]:
        yes_price = yes_df.iloc[i]["price"]
        no_price = no_df.iloc[i]["price"]
        total = yes_price + no_price
        timestamp = yes_df.iloc[i]["timestamp"]

        print(f"  {timestamp}: Yes={yes_price:.3f}, No={no_price:.3f}, Sum={total:.3f}")


def find_interesting_patterns():
    """Find interesting patterns in the data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Find Patterns")
    print("=" * 80)

    data = load_market_data("will-donald-trump-be-inaugurated")
    yes_df = data["outcomes"][0].copy()

    # Calculate returns
    yes_df["returns"] = yes_df["price"].pct_change()

    # Find largest price movements
    print("\nLargest price increases (10-minute intervals):")
    largest_increases = yes_df.nlargest(5, "returns")[["timestamp", "price", "returns"]]
    for idx, row in largest_increases.iterrows():
        print(
            f"  {row['timestamp']}: +{row['returns']*100:.2f}% (price: {row['price']:.3f})"
        )

    print("\nLargest price decreases (10-minute intervals):")
    largest_decreases = yes_df.nsmallest(5, "returns")[
        ["timestamp", "price", "returns"]
    ]
    for idx, row in largest_decreases.iterrows():
        print(
            f"  {row['timestamp']}: {row['returns']*100:.2f}% (price: {row['price']:.3f})"
        )

    # Find periods of high volatility
    yes_df["rolling_vol"] = (
        yes_df["returns"].rolling(window=144).std()
    )  # Daily rolling window

    print("\nHighest volatility periods (daily rolling):")
    high_vol = yes_df.nlargest(5, "rolling_vol")[["timestamp", "price", "rolling_vol"]]
    for idx, row in high_vol.iterrows():
        if pd.notna(row["rolling_vol"]):
            print(
                f"  {row['timestamp']}: vol={row['rolling_vol']:.4f} (price: {row['price']:.3f})"
            )


def export_data():
    """Export data for external analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Export Data")
    print("=" * 80)

    data = load_market_data("will-donald-trump-be-inaugurated")

    # Export to CSV for Excel/Pandas analysis
    yes_df = data["outcomes"][0]
    yes_df.to_csv("data/polymarket/trump_inauguration_yes.csv", index=False)
    print(f"  Exported Yes outcome to trump_inauguration_yes.csv")
    print(f"  {len(yes_df):,} rows")

    # Export summary statistics
    stats = yes_df["price"].describe()
    stats_df = pd.DataFrame({"metric": stats.index, "value": stats.values})
    stats_df.to_csv("data/polymarket/trump_inauguration_stats.csv", index=False)
    print(f"  Exported statistics to trump_inauguration_stats.csv")


def main():
    """Run all examples."""
    # Example 1: Download (will use cache if already downloaded)
    download_example()

    # Example 2: Load and analyze
    load_and_analyze()

    # Example 3: Compare outcomes
    compare_outcomes()

    # Example 4: Find patterns
    find_interesting_patterns()

    # Example 5: Export data
    export_data()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Try with different markets")
    print("  2. Create visualizations with plotly")
    print("  3. Build models using the price history")
    print("  4. Compare multiple markets")


if __name__ == "__main__":
    main()
