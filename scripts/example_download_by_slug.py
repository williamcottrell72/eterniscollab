"""
Example: Download Polymarket data using slugs instead of token IDs

This demonstrates two approaches:
1. For EVENTS (like presidential election): use download_polymarket_prices_by_event()
2. For individual MARKETS: use token IDs directly (market search can be slow)
"""

from datetime import datetime
from polymarket_data import (
    download_polymarket_prices_by_event,
    download_polymarket_prices,
    slug_to_token_ids,
)


def example_1_presidential_election():
    """
    Example 1: Download presidential election data using event slug

    This is the RECOMMENDED approach for events like the presidential election.
    """
    print("=" * 70)
    print("Example 1: Presidential Election (Event)")
    print("=" * 70)

    # Download Trump's data
    df_trump = download_polymarket_prices_by_event(
        event_slug="presidential-election-winner-2024",
        candidate_name="Donald Trump",
        outcome_index=0,  # "Yes" token
        start_date=datetime(2024, 10, 28),
        end_date=datetime(2024, 11, 5),
        fidelity=60,  # Hourly data
    )

    print(f"\nTrump - Downloaded {len(df_trump)} data points")
    if len(df_trump) > 0:
        print(df_trump.head())

    # Download Kamala Harris's data
    df_harris = download_polymarket_prices_by_event(
        event_slug="presidential-election-winner-2024",
        candidate_name="Kamala Harris",
        outcome_index=0,  # "Yes" token
        start_date=datetime(2024, 10, 28),
        end_date=datetime(2024, 11, 5),
        fidelity=60,  # Hourly data
    )

    print(f"\nHarris - Downloaded {len(df_harris)} data points")
    if len(df_harris) > 0:
        print(df_harris.head())


def example_2_list_all_candidates():
    """
    Example 2: List all candidates and their token IDs
    """
    print("\n" + "=" * 70)
    print("Example 2: List All Candidates")
    print("=" * 70)

    token_map = slug_to_token_ids("presidential-election-winner-2024", is_event=True)

    print(f"\nFound {len(token_map)} candidates:\n")
    for candidate, token_ids in token_map.items():
        print(f"  {candidate:30s} - Yes: {token_ids[0][:20]}...")


def example_3_direct_token_id():
    """
    Example 3: Download using token ID directly (fastest method)

    If you already know the token ID, this is the fastest approach.
    """
    print("\n" + "=" * 70)
    print("Example 3: Direct Token ID (Recommended for Speed)")
    print("=" * 70)

    # Trump's "Yes" token ID
    trump_token_id = (
        "21742633143463906290569050155826241533067272736897614950488156847949938836455"
    )

    df = download_polymarket_prices(
        token_id=trump_token_id,
        start_date=datetime(2024, 10, 28),
        end_date=datetime(2024, 11, 5),
        fidelity=60,  # Hourly data
    )

    print(f"\nDownloaded {len(df)} data points using token ID directly")
    if len(df) > 0:
        print(df.head())


if __name__ == "__main__":
    # Run all examples
    example_1_presidential_election()
    example_2_list_all_candidates()
    example_3_direct_token_id()

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
