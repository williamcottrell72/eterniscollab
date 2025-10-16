"""
Example: Download Presidential Election 2024 data from Polymarket

This script demonstrates how to:
1. Map from event slug to token IDs for all candidates
2. Download historical price data for specific candidates
"""

from datetime import datetime, timedelta
from polymarket_data import slug_to_token_ids, download_polymarket_prices
import json


def main():
    print("=" * 70)
    print("Polymarket Presidential Election 2024 - Token ID Lookup")
    print("=" * 70)

    # Get all token IDs for the presidential election event
    token_ids_by_candidate = slug_to_token_ids(
        "presidential-election-winner-2024", is_event=True
    )

    print(f"\nFound {len(token_ids_by_candidate)} candidates:\n")
    for candidate, token_ids in token_ids_by_candidate.items():
        print(f"{candidate:30s} - Yes: {token_ids[0]}")

    # Example: Download data for Donald Trump
    print("\n" + "=" * 70)
    print("Example: Downloading data for Donald Trump")
    print("=" * 70)

    trump_token_id = token_ids_by_candidate["Donald Trump"][0]  # "Yes" token

    # Download one week of data (hourly resolution to keep data size small)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=7)

    df = download_polymarket_prices(
        token_id=trump_token_id,
        start_date=start_date,
        end_date=end_date,
        fidelity=60,  # 60-minute bars (hourly)
        data_dir="data/polymarket",
    )

    print(f"\nDownloaded {len(df)} hourly data points")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())
    print(f"\nPrice statistics:")
    print(df["price"].describe())

    # Save token mapping to JSON file for reference
    output_file = "presidential_election_2024_tokens.json"
    with open(output_file, "w") as f:
        json.dump(token_ids_by_candidate, f, indent=2)
    print(f"\n\nToken ID mapping saved to: {output_file}")


if __name__ == "__main__":
    main()
