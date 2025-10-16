"""
Market History Downloader for Polymarket

This module provides functions to download complete price histories for markets
with all outcomes, organized in a convenient directory structure with metadata.

Features:
- Downloads all outcomes for each market
- Saves metadata alongside price data
- Implements rate limiting and retry logic
- Organizes data in market-specific folders
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from polymarket_data import (
    get_all_closed_markets,
    get_market_info,
    download_polymarket_prices,
)


class RateLimiter:
    """
    Simple rate limiter to avoid overloading Polymarket servers.

    Implements exponential backoff on errors and maintains a minimum
    delay between requests.
    """

    def __init__(self, min_delay: float = 0.5, max_delay: float = 60.0):
        """
        Initialize rate limiter.

        Args:
            min_delay: Minimum delay between requests in seconds (default: 0.5)
            max_delay: Maximum delay on exponential backoff (default: 60)
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0
        self.consecutive_errors = 0

    def wait(self):
        """Wait appropriate time before next request."""
        now = time.time()
        time_since_last = now - self.last_request_time

        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)

        self.last_request_time = time.time()

    def on_success(self):
        """Called when request succeeds - reset error count."""
        self.consecutive_errors = 0

    def on_error(self):
        """Called when request fails - increase backoff."""
        self.consecutive_errors += 1
        delay = min(self.min_delay * (2**self.consecutive_errors), self.max_delay)
        print(
            f"  Rate limiter: Backing off for {delay:.1f}s after error {self.consecutive_errors}"
        )
        time.sleep(delay)


def download_market_complete_history(
    market_slug: str,
    output_dir: str = "data/polymarket/market_histories",
    fidelity: int = 10,
    rate_limiter: Optional[RateLimiter] = None,
    overwrite: bool = False,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Download complete price history for all outcomes of a market.

    This function:
    1. Fetches market metadata
    2. Downloads price history for all outcomes
    3. Saves everything in organized directory structure:
       output_dir/
         market_slug/
           metadata.json       # Market information
           outcome_0.parquet   # Price history for outcome 0
           outcome_1.parquet   # Price history for outcome 1
           ...
           summary.json        # Summary stats

    Args:
        market_slug: The market slug (e.g., "will-donald-trump-be-inaugurated")
        output_dir: Base directory for organized market data
        fidelity: Data resolution in minutes (default: 10)
        rate_limiter: Optional RateLimiter instance (created if None)
        overwrite: If True, re-download even if data exists
        max_retries: Maximum number of retry attempts per outcome

    Returns:
        Dictionary with download summary:
        {
            'slug': str,
            'success': bool,
            'outcomes_downloaded': int,
            'total_data_points': int,
            'error': str (if failed)
        }

    Example:
        >>> result = download_market_complete_history(
        ...     "will-donald-trump-be-inaugurated",
        ...     fidelity=10
        ... )
        >>> print(f"Downloaded {result['outcomes_downloaded']} outcomes")
    """
    if rate_limiter is None:
        rate_limiter = RateLimiter()

    # Create market-specific directory
    market_dir = Path(output_dir) / market_slug
    market_dir.mkdir(parents=True, exist_ok=True)

    summary_file = market_dir / "summary.json"

    # Check if already downloaded
    if not overwrite and summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        if summary.get("success", False):
            print(f"  ✓ Already downloaded: {market_slug}")
            return summary

    print(f"\n{'='*80}")
    print(f"Downloading market: {market_slug}")
    print(f"{'='*80}")

    try:
        # Fetch market metadata with rate limiting
        rate_limiter.wait()
        print("  Fetching market metadata...")
        market_info = get_market_info(market_slug)
        rate_limiter.on_success()

        # Extract key information
        question = market_info.get("question", "Unknown")
        outcomes_str = market_info.get("outcomes", "[]")
        outcomes = json.loads(outcomes_str) if outcomes_str else []
        token_ids_str = market_info.get("clobTokenIds", "[]")
        token_ids = json.loads(token_ids_str) if token_ids_str else []

        # Get date range
        created_at_str = market_info.get("createdAt", market_info.get("startDate"))
        end_date_str = market_info.get("endDate", market_info.get("closedTime"))

        if not created_at_str or not end_date_str:
            return {
                "slug": market_slug,
                "success": False,
                "error": "Missing date information",
            }

        # Parse dates
        start_date = pd.to_datetime(created_at_str)
        end_date = pd.to_datetime(end_date_str)

        # Convert to datetime objects
        if hasattr(start_date, "to_pydatetime"):
            start_date = start_date.to_pydatetime()
        if hasattr(end_date, "to_pydatetime"):
            end_date = end_date.to_pydatetime()

        print(f"  Question: {question[:80]}...")
        print(f"  Outcomes: {outcomes}")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")
        print(f"  Duration: {(end_date - start_date).days} days")
        print(f"  Token IDs: {len(token_ids)}")

        # Save metadata
        metadata = {
            "slug": market_slug,
            "question": question,
            "outcomes": outcomes,
            "token_ids": token_ids,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "volume": market_info.get("volume", market_info.get("volumeNum", 0)),
            "liquidity": market_info.get(
                "liquidity", market_info.get("liquidityNum", 0)
            ),
            "category": market_info.get("category", "Unknown"),
            "full_info": market_info,
            "download_timestamp": datetime.now().isoformat(),
            "fidelity": fidelity,
        }

        with open(market_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved metadata to {market_dir / 'metadata.json'}")

        # Download price history for each outcome
        total_data_points = 0
        outcomes_downloaded = 0

        for outcome_idx, (outcome_name, token_id) in enumerate(
            zip(outcomes, token_ids)
        ):
            print(f"\n  Downloading outcome {outcome_idx}: {outcome_name}")
            print(f"    Token ID: {token_id[:32]}...")

            outcome_file = market_dir / f"outcome_{outcome_idx}.parquet"

            # Skip if already exists and not overwriting
            if not overwrite and outcome_file.exists():
                print(f"    ✓ Already exists, skipping")
                df = pd.read_parquet(outcome_file)
                total_data_points += len(df)
                outcomes_downloaded += 1
                continue

            # Retry logic for this outcome
            for attempt in range(max_retries):
                try:
                    rate_limiter.wait()

                    df = download_polymarket_prices(
                        token_id=token_id,
                        start_date=start_date,
                        end_date=end_date,
                        fidelity=fidelity,
                        data_dir=str(market_dir),
                        overwrite=overwrite,
                    )

                    # Save with meaningful name
                    df.to_parquet(outcome_file, index=False)
                    print(f"    ✓ Saved {len(df)} data points to {outcome_file.name}")

                    total_data_points += len(df)
                    outcomes_downloaded += 1
                    rate_limiter.on_success()
                    break

                except Exception as e:
                    print(f"    ✗ Attempt {attempt + 1}/{max_retries} failed: {e}")
                    rate_limiter.on_error()

                    if attempt == max_retries - 1:
                        print(
                            f"    ✗ Failed to download outcome {outcome_idx} after {max_retries} attempts"
                        )
                        # Continue to next outcome

        # Create summary
        summary = {
            "slug": market_slug,
            "question": question,
            "success": outcomes_downloaded > 0,
            "outcomes_downloaded": outcomes_downloaded,
            "total_outcomes": len(outcomes),
            "total_data_points": total_data_points,
            "download_timestamp": datetime.now().isoformat(),
            "fidelity": fidelity,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days,
            },
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"\n  ✓ COMPLETE: Downloaded {outcomes_downloaded}/{len(outcomes)} outcomes"
        )
        print(f"    Total data points: {total_data_points}")
        print(f"    Saved to: {market_dir}")

        return summary

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return {"slug": market_slug, "success": False, "error": str(e)}


def download_market_list(
    market_slugs: List[str],
    output_dir: str = "data/polymarket/market_histories",
    fidelity: int = 10,
    rate_limiter: Optional[RateLimiter] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Download complete histories for a list of markets.

    Args:
        market_slugs: List of market slugs to download
        output_dir: Base directory for organized market data
        fidelity: Data resolution in minutes (default: 10)
        rate_limiter: Optional RateLimiter instance
        overwrite: If True, re-download even if data exists

    Returns:
        Dictionary with overall summary:
        {
            'total_markets': int,
            'successful': int,
            'failed': int,
            'total_outcomes': int,
            'total_data_points': int,
            'results': [list of individual summaries]
        }

    Example:
        >>> slugs = ['market-1', 'market-2', 'market-3']
        >>> summary = download_market_list(slugs, fidelity=10)
        >>> print(f"Downloaded {summary['successful']}/{summary['total_markets']} markets")
    """
    if rate_limiter is None:
        rate_limiter = RateLimiter(min_delay=0.5, max_delay=60.0)

    print("=" * 80)
    print(f"DOWNLOADING {len(market_slugs)} MARKETS")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Fidelity: {fidelity} minutes")
    print(f"Overwrite: {overwrite}")
    print("=" * 80)

    results = []
    successful = 0
    failed = 0
    total_outcomes = 0
    total_data_points = 0

    start_time = time.time()

    for idx, slug in enumerate(market_slugs, 1):
        print(f"\n[{idx}/{len(market_slugs)}] Processing: {slug}")

        result = download_market_complete_history(
            market_slug=slug,
            output_dir=output_dir,
            fidelity=fidelity,
            rate_limiter=rate_limiter,
            overwrite=overwrite,
        )

        results.append(result)

        if result["success"]:
            successful += 1
            total_outcomes += result.get("outcomes_downloaded", 0)
            total_data_points += result.get("total_data_points", 0)
        else:
            failed += 1

    elapsed = time.time() - start_time

    # Overall summary
    summary = {
        "total_markets": len(market_slugs),
        "successful": successful,
        "failed": failed,
        "total_outcomes": total_outcomes,
        "total_data_points": total_data_points,
        "elapsed_time": elapsed,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    # Save overall summary
    summary_dir = Path(output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "download_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Total markets: {len(market_slugs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total outcomes downloaded: {total_outcomes}")
    print(f"Total data points: {total_data_points:,}")
    print(f"Elapsed time: {elapsed/60:.1f} minutes")
    print(f"Summary saved to: {summary_dir / 'download_summary.json'}")
    print("=" * 80)

    return summary


def load_market_data(
    market_slug: str, base_dir: str = "data/polymarket/market_histories"
) -> Dict[str, Any]:
    """
    Load all data for a market from disk.

    Args:
        market_slug: The market slug
        base_dir: Base directory where market data is stored

    Returns:
        Dictionary containing:
        {
            'metadata': dict,
            'summary': dict,
            'outcomes': [DataFrame for each outcome],
            'outcome_names': [names of outcomes]
        }

    Example:
        >>> data = load_market_data('will-donald-trump-be-inaugurated')
        >>> print(data['metadata']['question'])
        >>> for name, df in zip(data['outcome_names'], data['outcomes']):
        ...     print(f"{name}: {len(df)} data points")
    """
    market_dir = Path(base_dir) / market_slug

    if not market_dir.exists():
        raise ValueError(f"Market data not found: {market_dir}")

    # Load metadata
    with open(market_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Load summary
    with open(market_dir / "summary.json", "r") as f:
        summary = json.load(f)

    # Load outcome data
    outcomes = []
    outcome_files = sorted(market_dir.glob("outcome_*.parquet"))

    for outcome_file in outcome_files:
        df = pd.read_parquet(outcome_file)
        outcomes.append(df)

    return {
        "metadata": metadata,
        "summary": summary,
        "outcomes": outcomes,
        "outcome_names": metadata.get("outcomes", []),
    }


if __name__ == "__main__":
    # Example: Download a single market
    result = download_market_complete_history(
        "will-donald-trump-be-inaugurated", fidelity=10
    )

    print("\n" + "=" * 80)
    print("Example result:")
    print(json.dumps(result, indent=2))
