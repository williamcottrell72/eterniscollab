"""
Improved Workflow for Polymarket Data Download

This example demonstrates the recommended workflow:
1. Use get_event_markets() to explore available markets
2. Use download_polymarket_prices_by_event() to download data

This approach generalizes to ALL events, not just elections!
"""

from datetime import datetime
from polymarket_data import get_event_markets, download_polymarket_prices_by_event


def explore_event_markets(event_slug: str):
    """
    Step 1: Explore what markets are available in an event.
    """
    print("=" * 70)
    print(f"Exploring Event: {event_slug}")
    print("=" * 70)

    markets = get_event_markets(event_slug)

    print(f"\nFound {len(markets)} markets:\n")

    for market_id, info in markets.items():
        volume = float(info["volume"]) if info["volume"] else 0
        print(f"{market_id}")
        print(f"  Question: {info['question']}")
        print(f"  Outcomes: {info['outcomes']}")
        print(f"  Volume: ${volume:,.0f}")
        print(f"  Closed: {info['closed']}")
        print()

    return markets


def download_market_data(event_slug: str, market_id: str):
    """
    Step 2: Download data for a specific market.
    """
    print("=" * 70)
    print(f"Downloading: {event_slug} - {market_id}")
    print("=" * 70)

    df = download_polymarket_prices_by_event(
        event_slug=event_slug,
        market_id=market_id,
        outcome_index=0,  # "Yes" token
        start_date=datetime(2024, 10, 28),
        end_date=datetime(2024, 11, 5),
        fidelity=60,  # Hourly data
    )

    print(f"\nDownloaded {len(df)} data points")

    if len(df) > 0:
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nPrice statistics:")
        print(df["price"].describe())

    return df


def compare_multiple_markets(event_slug: str, market_ids: list):
    """
    Step 3: Download and compare multiple markets.
    """
    print("=" * 70)
    print(f"Comparing Markets in: {event_slug}")
    print("=" * 70)

    results = {}
    for market_id in market_ids:
        df = download_polymarket_prices_by_event(
            event_slug=event_slug,
            market_id=market_id,
            outcome_index=0,
            start_date=datetime(2024, 10, 28),
            end_date=datetime(2024, 11, 5),
            fidelity=60,
        )
        results[market_id] = df

    # Summary
    print(f"\n{'Market':<30} {'Data Points':<15} {'Avg Price':<15}")
    print("-" * 60)
    for market_id, df in results.items():
        if len(df) > 0:
            avg_price = df["price"].mean()
            print(f"{market_id:<30} {len(df):<15} {avg_price:.4f}")
        else:
            print(f"{market_id:<30} {len(df):<15} N/A")

    return results


if __name__ == "__main__":
    # Example 1: Explore presidential election markets
    markets = explore_event_markets("presidential-election-winner-2024")

    # Example 2: Download data for a specific candidate
    df_trump = download_market_data(
        event_slug="presidential-election-winner-2024", market_id="Donald Trump"
    )

    # Example 3: Compare multiple candidates
    compare_multiple_markets(
        event_slug="presidential-election-winner-2024",
        market_ids=["Donald Trump", "Kamala Harris", "Robert F. Kennedy Jr."],
    )

    print("\n" + "=" * 70)
    print("Done! This workflow generalizes to ANY Polymarket event.")
    print("=" * 70)
