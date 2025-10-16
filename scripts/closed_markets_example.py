"""
Example usage of get_all_closed_markets() function.

This script demonstrates how to fetch metadata for all closed Polymarket markets
and perform various analyses on the data.
"""

from polymarket_data import get_all_closed_markets
import pandas as pd


def main():
    """Demonstrate usage of get_all_closed_markets function."""

    # Fetch all closed markets (cached to disk)
    # First run will take a few minutes to download all markets
    # Subsequent runs will load from cache instantly
    print("Fetching all closed markets metadata...")
    print("(First run may take a few minutes, subsequent runs use cache)")

    df = get_all_closed_markets(
        cache_dir="data/polymarket", overwrite=False  # Use cached data if available
    )

    print(f"\n✓ Loaded {len(df)} closed markets")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Cache location: data/polymarket/closed_markets_metadata.parquet")

    # Display basic statistics
    print("\n" + "=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)

    print(f"\nVolume Statistics:")
    print(f"  Total volume: ${df['volume_num'].sum():,.2f}")
    print(f"  Mean volume: ${df['volume_num'].mean():,.2f}")
    print(f"  Median volume: ${df['volume_num'].median():,.2f}")
    print(f"  Max volume: ${df['volume_num'].max():,.2f}")

    print(f"\nLiquidity Statistics:")
    print(f"  Mean liquidity: ${df['liquidity_num'].mean():,.2f}")
    print(f"  Median liquidity: ${df['liquidity_num'].median():,.2f}")

    # Date range
    valid_end_dates = df["end_date_parsed"].dropna()
    if len(valid_end_dates) > 0:
        print(f"\nDate Range:")
        print(f"  Earliest market ended: {valid_end_dates.min()}")
        print(f"  Latest market ended: {valid_end_dates.max()}")

    # Category breakdown
    print("\n" + "=" * 80)
    print("TOP 10 CATEGORIES")
    print("=" * 80)
    category_counts = df["category"].value_counts().head(10)
    for category, count in category_counts.items():
        print(f"  {category}: {count} markets")

    # Top markets by volume
    print("\n" + "=" * 80)
    print("TOP 10 MARKETS BY VOLUME")
    print("=" * 80)
    top_volume = df.nlargest(10, "volume_num")[
        ["question", "volume_num", "category", "slug"]
    ]
    for idx, row in top_volume.iterrows():
        print(f"\n  ${row['volume_num']:,.0f} - {row['category']}")
        print(f"  {row['question'][:80]}...")
        print(f"  Slug: {row['slug']}")

    # Filter examples
    print("\n" + "=" * 80)
    print("FILTER EXAMPLES")
    print("=" * 80)

    # High-volume markets (>$1M)
    high_volume = df[df["volume_num"] > 1_000_000]
    print(f"\nMarkets with >$1M volume: {len(high_volume)}")

    # High-liquidity markets
    high_liquidity = df[df["liquidity_num"] > 1000]
    print(f"Markets with >$1000 liquidity: {len(high_liquidity)}")

    # Recent markets (ended in last year)
    recent_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
    recent_markets = df[df["end_date_parsed"] > recent_cutoff]
    print(f"Markets that ended in last 365 days: {len(recent_markets)}")

    # Politics category
    politics = df[df["category"].str.lower().str.contains("politics", na=False)]
    print(f"Politics markets: {len(politics)}")

    # Combine filters: high-volume politics markets
    high_vol_politics = df[
        (df["volume_num"] > 500_000)
        & (df["category"].str.lower().str.contains("politics", na=False))
    ]
    print(f"\nHigh-volume (>$500k) politics markets: {len(high_vol_politics)}")
    if len(high_vol_politics) > 0:
        print("\nTop 5:")
        for idx, row in high_vol_politics.nlargest(5, "volume_num").iterrows():
            print(f"  ${row['volume_num']:,.0f} - {row['question'][:60]}...")

    # Export example: save high-volume markets to CSV
    print("\n" + "=" * 80)
    print("EXPORT EXAMPLE")
    print("=" * 80)

    output_file = "data/polymarket/high_volume_markets.csv"
    high_volume_export = df[df["volume_num"] > 1_000_000][
        [
            "slug",
            "question",
            "volume_num",
            "liquidity_num",
            "category",
            "end_date_parsed",
            "outcomes",
            "outcomePrices",
        ]
    ]
    high_volume_export.to_csv(output_file, index=False)
    print(f"\nExported {len(high_volume_export)} high-volume markets to:")
    print(f"  {output_file}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Explore the data: df = get_all_closed_markets()")
    print("  2. Filter for markets of interest (by volume, category, date, etc.)")
    print(
        "  3. Use download_polymarket_prices() to get price history for specific markets"
    )
    print("  4. Analyze price movements, outcomes, and market efficiency")


if __name__ == "__main__":
    main()
