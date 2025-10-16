"""
Download price histories for selected high, medium, and low volume markets.

This script downloads complete price histories for 30 markets:
- 10 high volume markets
- 10 medium volume markets
- 10 low volume markets

Results are saved in organized directory structure for later analysis.
"""

import json
from market_history_downloader import download_market_list, RateLimiter


def main():
    """Download all selected markets."""

    # Load selected market slugs
    with open("data/polymarket/selected_markets.json", "r") as f:
        selected = json.load(f)

    # Combine all markets
    all_slugs = (
        selected["high_volume"] + selected["medium_volume"] + selected["low_volume"]
    )

    print(f"Selected markets:")
    print(f"  High volume: {len(selected['high_volume'])}")
    print(f"  Medium volume: {len(selected['medium_volume'])}")
    print(f"  Low volume: {len(selected['low_volume'])}")
    print(f"  Total: {len(all_slugs)}")

    # Create rate limiter with conservative settings
    # 0.5s minimum delay between requests
    # Up to 60s backoff on errors
    rate_limiter = RateLimiter(min_delay=0.5, max_delay=60.0)

    # Download all markets
    summary = download_market_list(
        market_slugs=all_slugs,
        output_dir="data/polymarket/market_histories",
        fidelity=10,  # 10-minute bars
        rate_limiter=rate_limiter,
        overwrite=False,  # Don't re-download existing data
    )

    # Print detailed summary
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    print("\nSuccessful downloads:")
    for result in summary["results"]:
        if result["success"]:
            print(f"  ✓ {result['slug']}")
            print(
                f"    Outcomes: {result.get('outcomes_downloaded', 0)}/{result.get('total_outcomes', 0)}"
            )
            print(f"    Data points: {result.get('total_data_points', 0):,}")

    if summary["failed"] > 0:
        print("\nFailed downloads:")
        for result in summary["results"]:
            if not result["success"]:
                print(f"  ✗ {result['slug']}")
                print(f"    Error: {result.get('error', 'Unknown')}")

    print("\n" + "=" * 80)
    print("SUMMARY BY VOLUME TIER")
    print("=" * 80)

    # Group results by tier
    high_results = [r for r in summary["results"][:10]]
    medium_results = [r for r in summary["results"][10:20]]
    low_results = [r for r in summary["results"][20:30]]

    for tier_name, tier_results in [
        ("HIGH VOLUME", high_results),
        ("MEDIUM VOLUME", medium_results),
        ("LOW VOLUME", low_results),
    ]:
        successful = sum(1 for r in tier_results if r["success"])
        total_outcomes = sum(r.get("outcomes_downloaded", 0) for r in tier_results)
        total_points = sum(r.get("total_data_points", 0) for r in tier_results)

        print(f"\n{tier_name}:")
        print(f"  Markets: {successful}/10 successful")
        print(f"  Outcomes: {total_outcomes}")
        print(f"  Data points: {total_points:,}")


if __name__ == "__main__":
    main()
