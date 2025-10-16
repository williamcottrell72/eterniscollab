"""
Polymarket Data Downloader

This module provides functions to download and cache historical price data from Polymarket's CLOB API.

Key features:
- Download minute-level price data for any market token
- Automatic caching to disk to avoid redundant API calls
- Support for date ranges and time intervals
- Integration with Polymarket's Gamma API for market discovery

API Documentation:
- CLOB API: https://docs.polymarket.com/developers/CLOB/timeseries
- Gamma Markets API: https://docs.polymarket.com/developers/gamma-markets-api/overview
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path
import pandas as pd


# API Base URLs
CLOB_API_BASE = "https://clob.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


def get_event_info(event_slug: str) -> Dict[str, Any]:
    """
    Get event information from Polymarket's Gamma Events API.

    Events are collections of related markets. For example, the "presidential-election-winner-2024"
    event contains separate markets for each candidate (Trump, Biden, Harris, etc.).

    Args:
        event_slug: The event slug (e.g., "presidential-election-winner-2024")

    Returns:
        Dictionary containing event information and all associated markets

    Raises:
        ValueError: If event not found
        requests.RequestException: If API request fails

    Example:
        >>> event = get_event_info("presidential-election-winner-2024")
        >>> print(event['title'])
        >>> for market in event['markets']:
        ...     print(market['question'], market['clobTokenIds'])
    """
    url = f"{GAMMA_API_BASE}/events"

    response = requests.get(url, params={"slug": event_slug})
    response.raise_for_status()

    events = response.json()

    if not events or len(events) == 0:
        raise ValueError(f"Event with slug '{event_slug}' not found")

    # The API returns a list with one event when querying by slug
    return events[0]


def get_market_info(market_slug: str, max_pages: int = 10) -> Dict[str, Any]:
    """
    Get market information from Polymarket's Gamma API.

    This function first tries to query directly by slug (fast), then falls back
    to pagination if the direct query fails.

    Args:
        market_slug: The market slug (e.g., "fed-rate-hike-in-2025")
        max_pages: Maximum number of pages to search (default: 10, i.e., 10,000 markets)
                   Only used if direct slug query fails

    Returns:
        Dictionary containing market information including token IDs

    Raises:
        ValueError: If market not found
        requests.RequestException: If API request fails
    """
    url = f"{GAMMA_API_BASE}/markets"

    # First, try direct query by slug (fast)
    response = requests.get(url, params={"slug": market_slug})
    response.raise_for_status()
    markets = response.json()

    if markets and len(markets) > 0:
        # Direct query succeeded
        return markets[0]

    # Fallback: Paginate through markets (slow)
    print(f"Direct slug query failed, falling back to pagination (this may be slow)...")
    for page in range(max_pages):
        offset = page * 1000
        response = requests.get(url, params={"limit": 1000, "offset": offset})
        response.raise_for_status()

        markets = response.json()

        if not markets:  # No more markets
            break

        for market in markets:
            if market.get("slug") == market_slug:
                return market

    raise ValueError(f"Market with slug '{market_slug}' not found")


def slug_to_token_ids(slug: str, is_event: bool = False) -> Dict[str, List[str]]:
    """
    Map a Polymarket slug to token IDs.

    This function handles both individual markets and events (collections of markets).
    For events like "presidential-election-winner-2024", it returns token IDs for all
    associated markets.

    Args:
        slug: The market or event slug (e.g., "presidential-election-winner-2024")
        is_event: If True, treat slug as an event slug; if False, treat as market slug

    Returns:
        Dictionary mapping market questions/names to their token ID lists.
        For simple markets: {"market_question": ["token_id_yes", "token_id_no"]}
        For events: {"Trump": ["token_id_yes", "token_id_no"], "Biden": [...], ...}

    Raises:
        ValueError: If slug not found
        requests.RequestException: If API request fails

    Examples:
        >>> # For an event (multiple markets)
        >>> result = slug_to_token_ids("presidential-election-winner-2024", is_event=True)
        >>> print(result.keys())  # ['Donald Trump', 'Joe Biden', 'Kamala Harris', ...]
        >>> trump_tokens = result['Donald Trump']
        >>> print(trump_tokens[0])  # Token ID for "Yes" on Trump

        >>> # For a single market
        >>> result = slug_to_token_ids("fed-rate-hike-in-2025", is_event=False)
        >>> print(result)  # {'Will the Fed...': ['token_yes', 'token_no']}
    """
    if is_event:
        # Get event info
        event = get_event_info(slug)

        result = {}
        for market in event.get("markets", []):
            # Parse token IDs
            token_ids_str = market.get("clobTokenIds", "")
            if token_ids_str:
                token_ids = json.loads(token_ids_str)

                # Use groupItemTitle if available (e.g., "Donald Trump"), otherwise use question
                market_name = market.get(
                    "groupItemTitle", market.get("question", "Unknown")
                )
                result[market_name] = token_ids

        return result
    else:
        # Get individual market info
        market = get_market_info(slug)

        token_ids_str = market.get("clobTokenIds", "")
        if not token_ids_str:
            raise ValueError(f"No CLOB token IDs found for market '{slug}'")

        token_ids = json.loads(token_ids_str)
        question = market.get("question", slug)

        return {question: token_ids}


def download_polymarket_prices(
    token_id: str,
    start_date: datetime,
    end_date: datetime,
    fidelity: int = 1,
    data_dir: str = "data/polymarket",
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download historical price data for a Polymarket token.

    This function downloads minute-level (or other resolution) price data from Polymarket's
    CLOB API and caches it locally. If the data already exists on disk and overwrite=False,
    it will load from cache instead of making an API call.

    IMPORTANT: The Polymarket API has a ~15 day limit per request. For longer date ranges,
    this function automatically chunks the requests and combines the results.

    Args:
        token_id: The CLOB token ID (large integer as string)
        start_date: Start date for historical data
        end_date: End date for historical data
        fidelity: Data resolution in minutes (default: 1 for minute bars)
        data_dir: Directory to store cached data (default: "data/polymarket")
        overwrite: If True, download even if cached data exists (default: False)

    Returns:
        DataFrame with columns: timestamp (datetime), price (float), unix_timestamp (int)

    Raises:
        requests.RequestException: If API request fails
        ValueError: If invalid parameters provided

    Example:
        >>> df = download_polymarket_prices(
        ...     token_id="60487116984468020978247225474488676749601001829886755968952521846780452448915",
        ...     start_date=datetime(2024, 10, 28),
        ...     end_date=datetime(2024, 10, 29),
        ...     fidelity=1
        ... )
        >>> print(df.head())
    """
    # Validate inputs
    if start_date >= end_date:
        raise ValueError("start_date must be before end_date")

    if fidelity < 1:
        raise ValueError("fidelity must be at least 1 minute")

    # Check if date range exceeds API limit (~15 days)
    days_requested = (end_date - start_date).days
    MAX_DAYS_PER_REQUEST = 14  # Conservative limit (API supports ~15)

    if days_requested > MAX_DAYS_PER_REQUEST:
        # Split into chunks and download separately
        print(
            f"Date range ({days_requested} days) exceeds API limit ({MAX_DAYS_PER_REQUEST} days)"
        )
        print(
            f"Splitting into {(days_requested + MAX_DAYS_PER_REQUEST - 1) // MAX_DAYS_PER_REQUEST} chunks..."
        )

        chunks = []
        current_start = start_date

        while current_start < end_date:
            chunk_end = min(
                current_start + timedelta(days=MAX_DAYS_PER_REQUEST), end_date
            )

            print(f"  Downloading chunk: {current_start.date()} to {chunk_end.date()}")
            chunk_df = download_polymarket_prices(
                token_id=token_id,
                start_date=current_start,
                end_date=chunk_end,
                fidelity=fidelity,
                data_dir=data_dir,
                overwrite=overwrite,
            )

            if len(chunk_df) > 0:
                chunks.append(chunk_df)

            current_start = chunk_end

        if not chunks:
            print("Warning: No data returned from API")
            return pd.DataFrame(columns=["timestamp", "price", "unix_timestamp"])

        # Combine all chunks
        combined_df = pd.concat(chunks, ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=["unix_timestamp"]
        ).reset_index(drop=True)
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        print(
            f"Combined {len(chunks)} chunks into {len(combined_df)} total data points"
        )
        return combined_df

    # Create data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Generate cache filename based on parameters
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    cache_filename = (
        f"token_{token_id[:16]}_{start_str}_{end_str}_fid{fidelity}.parquet"
    )
    cache_path = data_path / cache_filename

    # Check cache if not overwriting
    if not overwrite and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    # Convert dates to Unix timestamps
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    # Make API request
    url = f"{CLOB_API_BASE}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity,
    }

    print(f"Downloading data from Polymarket API...")
    print(f"  Token ID: {token_id[:32]}...")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print(f"  Fidelity: {fidelity} minute(s)")

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    # Parse response
    if "history" not in data:
        raise ValueError(f"Unexpected API response format: {data}")

    history = data["history"]

    if not history:
        print("Warning: No data returned from API")
        return pd.DataFrame(columns=["timestamp", "price", "unix_timestamp"])

    # Convert to DataFrame
    df = pd.DataFrame(history)
    df.columns = ["unix_timestamp", "price"]

    # Convert Unix timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["unix_timestamp"], unit="s")

    # Reorder columns
    df = df[["timestamp", "price", "unix_timestamp"]]

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter to requested date range (API sometimes returns data outside range)
    # Use Unix timestamps for filtering to avoid datetime comparison issues
    start_ts_unix = int(start_date.timestamp())
    end_ts_unix = int(end_date.timestamp())
    df = df[
        (df["unix_timestamp"] >= start_ts_unix) & (df["unix_timestamp"] < end_ts_unix)
    ].reset_index(drop=True)

    # Save to cache
    df.to_parquet(cache_path, index=False)
    print(f"Cached data saved to {cache_path}")
    print(f"Downloaded {len(df)} data points")

    return df


def download_polymarket_prices_by_slug(
    market_slug: str,
    outcome_index: int,
    start_date: datetime,
    end_date: datetime,
    fidelity: int = 1,
    data_dir: str = "data/polymarket",
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download historical price data for a Polymarket market by slug.

    This is a convenience wrapper around download_polymarket_prices that first looks up
    the token ID from a market slug.

    Args:
        market_slug: The market slug (e.g., "fed-rate-hike-in-2025")
        outcome_index: Index of the outcome (0 for first outcome, 1 for second, etc.)
        start_date: Start date for historical data
        end_date: End date for historical data
        fidelity: Data resolution in minutes (default: 1 for minute bars)
        data_dir: Directory to store cached data (default: "data/polymarket")
        overwrite: If True, download even if cached data exists (default: False)

    Returns:
        DataFrame with columns: timestamp (datetime), price (float), unix_timestamp (int)

    Example:
        >>> # Download "Yes" outcome for Fed rate hike market
        >>> df = download_polymarket_prices_by_slug(
        ...     market_slug="fed-rate-hike-in-2025",
        ...     outcome_index=0,
        ...     start_date=datetime(2024, 12, 1),
        ...     end_date=datetime(2024, 12, 2)
        ... )
    """
    # Get market info
    market_info = get_market_info(market_slug)

    # Extract token IDs
    token_ids_str = market_info.get("clobTokenIds", "")
    if not token_ids_str:
        raise ValueError(f"No CLOB token IDs found for market '{market_slug}'")

    token_ids = json.loads(token_ids_str)

    if outcome_index >= len(token_ids):
        raise ValueError(
            f"outcome_index {outcome_index} out of range. "
            f"Market has {len(token_ids)} outcomes."
        )

    token_id = token_ids[outcome_index]

    # Get outcomes for reference
    outcomes_str = market_info.get("outcomes", "")
    if outcomes_str:
        outcomes = json.loads(outcomes_str)
        print(f"Market: {market_info['question']}")
        print(f"Downloading outcome {outcome_index}: {outcomes[outcome_index]}")

    # Download data
    return download_polymarket_prices(
        token_id=token_id,
        start_date=start_date,
        end_date=end_date,
        fidelity=fidelity,
        data_dir=data_dir,
        overwrite=overwrite,
    )


def get_event_markets(event_slug: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all markets within an event with their metadata and token IDs.

    This function retrieves detailed information about all markets in an event,
    making it easy to explore what markets are available before downloading data.

    Args:
        event_slug: The event slug (e.g., "presidential-election-winner-2024")

    Returns:
        Dictionary mapping market identifiers to market metadata:
        {
            "market_identifier": {
                "question": "Full question text",
                "token_ids": ["yes_token", "no_token"],
                "slug": "market-slug",
                "description": "Market description",
                ...
            },
            ...
        }

        The market identifier is typically:
        - groupItemTitle (e.g., "Donald Trump") for grouped markets
        - question text for standalone markets

    Example:
        >>> markets = get_event_markets("presidential-election-winner-2024")
        >>> for market_id, info in markets.items():
        ...     print(f"{market_id}: {info['question']}")
        ...     print(f"  Token IDs: {info['token_ids']}")
        >>>
        >>> # Download data for a specific market
        >>> trump_tokens = markets["Donald Trump"]["token_ids"]
    """
    event = get_event_info(event_slug)

    result = {}
    for market in event.get("markets", []):
        # Determine market identifier (groupItemTitle for grouped markets, question otherwise)
        market_id = market.get("groupItemTitle", market.get("question", "Unknown"))

        # Parse token IDs
        token_ids_str = market.get("clobTokenIds", "")
        token_ids = json.loads(token_ids_str) if token_ids_str else []

        # Build market info
        result[market_id] = {
            "question": market.get("question", ""),
            "token_ids": token_ids,
            "slug": market.get("slug", ""),
            "description": market.get("description", ""),
            "outcomes": json.loads(market.get("outcomes", "[]")),
            "volume": market.get("volume", 0),
            "closed": market.get("closed", False),
            "end_date": market.get("endDate", ""),
        }

    return result


def download_polymarket_prices_by_event(
    event_slug: str,
    market_id: str,
    outcome_index: int = 0,
    start_date: datetime = None,
    end_date: datetime = None,
    fidelity: int = 1,
    data_dir: str = "data/polymarket",
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download historical price data for a specific market within an event.

    This function is for events (collections of markets), like the presidential election.
    Use get_event_markets() first to see available markets.

    Args:
        event_slug: The event slug (e.g., "presidential-election-winner-2024")
        market_id: Market identifier from get_event_markets()
                   (e.g., "Donald Trump", "Kamala Harris")
        outcome_index: 0 for "Yes" token, 1 for "No" token (default: 0)
        start_date: Start date for historical data
        end_date: End date for historical data
        fidelity: Data resolution in minutes (default: 1 for minute bars)
        data_dir: Directory to store cached data (default: "data/polymarket")
        overwrite: If True, download even if cached data exists (default: False)

    Returns:
        DataFrame with columns: timestamp (datetime), price (float), unix_timestamp (int)

    Raises:
        ValueError: If event not found or market_id not found in event

    Example:
        >>> # List available markets first
        >>> markets = get_event_markets("presidential-election-winner-2024")
        >>> print(list(markets.keys()))
        >>>
        >>> # Download Trump's odds
        >>> df = download_polymarket_prices_by_event(
        ...     event_slug="presidential-election-winner-2024",
        ...     market_id="Donald Trump",
        ...     outcome_index=0,  # "Yes" token
        ...     start_date=datetime(2024, 10, 28),
        ...     end_date=datetime(2024, 11, 5),
        ...     fidelity=60  # Hourly data
        ... )
    """
    # Get all markets for this event
    markets = get_event_markets(event_slug)

    # Find the market
    if market_id not in markets:
        available = ", ".join(markets.keys())
        raise ValueError(
            f"Market '{market_id}' not found in event '{event_slug}'. "
            f"Available markets: {available}"
        )

    market_info = markets[market_id]
    token_ids = market_info["token_ids"]

    if outcome_index >= len(token_ids):
        raise ValueError(
            f"outcome_index {outcome_index} out of range. "
            f"This market has {len(token_ids)} outcomes."
        )

    token_id = token_ids[outcome_index]

    outcome_name = "Yes" if outcome_index == 0 else "No"
    print(f"Event: {event_slug}")
    print(f"Market: {market_id}")
    print(f"Downloading {outcome_name} token")

    # Download data
    return download_polymarket_prices(
        token_id=token_id,
        start_date=start_date,
        end_date=end_date,
        fidelity=fidelity,
        data_dir=data_dir,
        overwrite=overwrite,
    )


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string columns that should be numeric to float type.

    This function detects columns that contain numeric data stored as strings
    and converts them to float. It handles common numeric column patterns like
    volume, liquidity, prices, and other financial metrics.

    Args:
        df: DataFrame with potentially mis-typed numeric columns

    Returns:
        DataFrame with numeric columns properly typed as float
    """
    # Common patterns for numeric column names (case-insensitive)
    numeric_column_patterns = [
        "volume",
        "liquidity",
        "price",
        "amount",
        "value",
        "num",
        "count",
        "total",
        "sum",
        "avg",
        "mean",
        "min",
        "max",
        "spread",
        "fee",
        "cost",
        "reward",
        "stake",
        "bet",
    ]

    # Track conversions for reporting
    converted_cols = []

    for col in df.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Check if column name matches numeric patterns
        col_lower = col.lower()
        is_numeric_col = any(
            pattern in col_lower for pattern in numeric_column_patterns
        )

        if is_numeric_col:
            # Try to convert to numeric
            try:
                # Use pd.to_numeric with errors='coerce' to handle invalid values
                converted = pd.to_numeric(df[col], errors="coerce")

                # Only apply conversion if we successfully converted at least some values
                non_null_count = converted.notna().sum()
                if non_null_count > 0:
                    df[col] = converted
                    converted_cols.append(col)
            except Exception:
                # Skip columns that can't be converted
                pass

    if converted_cols:
        print(
            f"  Converted {len(converted_cols)} columns to numeric: {', '.join(converted_cols[:5])}"
            + (f" and {len(converted_cols)-5} more" if len(converted_cols) > 5 else "")
        )

    return df


def get_all_closed_markets(
    cache_dir: str = "data/polymarket",
    overwrite: bool = False,
    max_markets: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch metadata for all closed Polymarket markets and cache to disk.

    This function retrieves comprehensive metadata for closed markets including
    slug, outcomes, volume, liquidity, end date, and other useful fields.
    Results are cached as a Parquet file for fast reloading.

    Args:
        cache_dir: Directory to store cached metadata (default: "data/polymarket")
        overwrite: If True, fetch fresh data even if cache exists (default: False)
        max_markets: Maximum number of markets to fetch (default: None = all markets)
                    Useful for testing with smaller datasets

    Returns:
        pandas DataFrame with columns:
            - id: Market ID
            - slug: Market slug
            - question: Market question text
            - description: Detailed description
            - outcomes: JSON string of possible outcomes
            - outcome_prices: Current outcome prices
            - closed: Whether market is closed
            - end_date: Market end date
            - end_date_iso: ISO format end date
            - volume: Trading volume
            - volume_num: Volume as float
            - liquidity: Market liquidity
            - liquidity_num: Liquidity as float
            - clob_token_ids: Token IDs for CLOB
            - category: Market category
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - market_type: Type of market
            And many other fields from the Polymarket API

    Raises:
        requests.RequestException: If API request fails

    Examples:
        >>> # Fetch all closed markets (uses cache if available)
        >>> df = get_all_closed_markets()
        >>> print(f"Found {len(df)} closed markets")
        >>> print(df[['slug', 'question', 'volume_num', 'end_date']].head())

        >>> # Force refresh from API
        >>> df = get_all_closed_markets(overwrite=True)

        >>> # Get just 100 markets for testing
        >>> df_sample = get_all_closed_markets(max_markets=100)

        >>> # Find high-volume markets
        >>> high_volume = df[df['volume_num'] > 100000].sort_values('volume_num', ascending=False)
        >>> print(high_volume[['slug', 'question', 'volume_num']].head(10))

        >>> # Find markets by category
        >>> election_markets = df[df['category'] == 'politics']
    """
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Define cache file
    cache_file = cache_path / "closed_markets_metadata.parquet"

    # Check cache
    if not overwrite and cache_file.exists():
        print(f"Loading closed markets metadata from cache: {cache_file}")
        df = pd.read_parquet(cache_file)
        print(f"Loaded {len(df)} closed markets from cache")

        # Convert numeric columns (in case cache has incorrect types)
        df = _convert_numeric_columns(df)

        if max_markets and len(df) > max_markets:
            df = df.head(max_markets)
            print(f"Limiting to first {max_markets} markets")

        return df

    # Fetch from API
    print("Fetching closed markets from Polymarket Gamma API...")
    print("This may take several minutes as we paginate through all markets...")

    all_markets = []
    offset = 0
    limit = 1000
    page = 1

    url = f"{GAMMA_API_BASE}/markets"

    while True:
        if max_markets and len(all_markets) >= max_markets:
            break

        print(f"  Fetching page {page} (offset={offset}, limit={limit})...")

        try:
            response = requests.get(
                url,
                params={
                    "closed": "true",  # Only closed markets
                    "limit": limit,
                    "offset": offset,
                },
                timeout=30,
            )
            response.raise_for_status()

            markets = response.json()

            if not markets:
                print(f"  No more markets found. Stopping at page {page}.")
                break

            print(f"  Retrieved {len(markets)} markets")
            all_markets.extend(markets)

            offset += limit
            page += 1

        except requests.exceptions.Timeout:
            print(f"  Timeout on page {page}. Retrying...")
            continue
        except requests.exceptions.RequestException as e:
            print(f"  Error on page {page}: {e}")
            if all_markets:
                print(f"  Stopping with {len(all_markets)} markets collected so far")
                break
            else:
                raise

    print(f"\nFetched {len(all_markets)} total closed markets")

    # Convert to DataFrame
    if not all_markets:
        print("Warning: No markets found")
        return pd.DataFrame()

    df = pd.DataFrame(all_markets)

    # Add parsed numeric columns for easier filtering
    if "volumeNum" in df.columns:
        df["volume_num"] = df["volumeNum"]
    elif "volume" in df.columns:
        df["volume_num"] = pd.to_numeric(df["volume"], errors="coerce")

    if "liquidityNum" in df.columns:
        df["liquidity_num"] = df["liquidityNum"]
    elif "liquidity" in df.columns:
        df["liquidity_num"] = pd.to_numeric(df["liquidity"], errors="coerce")

    # Parse dates
    if "endDate" in df.columns:
        df["end_date_parsed"] = pd.to_datetime(df["endDate"], errors="coerce")

    if "createdAt" in df.columns:
        df["created_at_parsed"] = pd.to_datetime(df["createdAt"], errors="coerce")

    if "closedTime" in df.columns:
        df["closed_time_parsed"] = pd.to_datetime(df["closedTime"], errors="coerce")

    # Limit if requested
    if max_markets and len(df) > max_markets:
        df = df.head(max_markets)
        print(f"Limiting to first {max_markets} markets")

    # Convert numeric columns to ensure proper typing
    df = _convert_numeric_columns(df)

    # Save to cache
    df.to_parquet(cache_file, index=False)
    print(f"\nCached {len(df)} markets to {cache_file}")
    print(f"Cache file size: {cache_file.stat().st_size / (1024*1024):.2f} MB")

    return df


def download_month_of_data(
    token_id: str,
    year: int,
    month: int,
    fidelity: int = 1,
    data_dir: str = "data/polymarket",
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download one month of historical price data for a Polymarket token.

    This is a convenience function that automatically determines the start and end
    dates for a given month.

    Args:
        token_id: The CLOB token ID (large integer as string)
        year: Year (e.g., 2024)
        month: Month (1-12)
        fidelity: Data resolution in minutes (default: 1 for minute bars)
        data_dir: Directory to store cached data (default: "data/polymarket")
        overwrite: If True, download even if cached data exists (default: False)

    Returns:
        DataFrame with columns: timestamp (datetime), price (float), unix_timestamp (int)

    Example:
        >>> # Download October 2024 data
        >>> df = download_month_of_data(
        ...     token_id="60487116984468020978247225474488676749601001829886755968952521846780452448915",
        ...     year=2024,
        ...     month=10
        ... )
    """
    # Validate month
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")

    # Calculate start and end dates for the month
    start_date = datetime(year, month, 1)

    # Calculate last day of month
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)

    return download_polymarket_prices(
        token_id=token_id,
        start_date=start_date,
        end_date=end_date,
        fidelity=fidelity,
        data_dir=data_dir,
        overwrite=overwrite,
    )


def load_all_market_histories(
    data_dir: str = "data/polymarket/market_histories",
    sample_per_day: bool = True,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Load all market histories and combine into a single DataFrame.

    For each slug and outcome in market_histories, this function loads the price data
    and metadata, optionally samples one row per date, and combines everything into
    a single DataFrame suitable for modeling.

    Args:
        data_dir: Directory containing market history folders
        sample_per_day: If True, sample one row per date (default: True)
                       If False, return all available data points
        date_column: Name for the date column (default: "date")

    Returns:
        DataFrame with columns:
            - slug: Market slug
            - outcome: Outcome label
            - timestamp: Original timestamp
            - date: Date (if sample_per_day=True)
            - price: Price at that timestamp
            - days_remaining: Days until market closure
            - resolution_price: Market resolution (0 or 1) from metadata
            - final_price: Market resolution (0 or 1) if available, else last observed price
            - avg_daily_volume: Average daily volume (total volume / days market was open)
            - question: Market question
            - volume: Trading volume (numeric)
            - liquidity: Market liquidity (numeric)
            - category: Market category
            - end_date: Market end date
            - closed: Whether market is closed
            - ... (additional numeric columns from metadata)

    Example:
        >>> # Load all markets with one sample per day
        >>> df = load_all_market_histories()
        >>> print(f"Loaded {len(df)} rows across {df['slug'].nunique()} markets")
        >>>
        >>> # Group by market and analyze
        >>> by_market = df.groupby('slug').agg({
        ...     'price': ['mean', 'std'],
        ...     'volume': 'first',
        ...     'days_remaining': ['min', 'max']
        ... })
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    all_data = []

    # Get all market directories
    market_dirs = [
        d for d in data_path.iterdir() if d.is_dir() and (d / "metadata.json").exists()
    ]

    if not market_dirs:
        print(f"Warning: No market directories found in {data_dir}")
        return pd.DataFrame()

    print(f"Loading {len(market_dirs)} markets from {data_dir}...")

    for market_dir in market_dirs:
        slug = market_dir.name

        # Load metadata
        metadata_path = market_dir / "metadata.json"
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"  Warning: Failed to load metadata for {slug}: {e}")
            continue

        # Parse end date for days_remaining calculation
        end_date_str = metadata.get("end_date")
        if end_date_str:
            end_date = pd.to_datetime(end_date_str)
        else:
            end_date = None

        # Get outcomes and resolution prices
        outcomes = metadata.get("outcomes", [])

        # Get outcome prices (resolution values: 0 or 1)
        full_info = metadata.get("full_info", {})
        outcome_prices_str = full_info.get("outcomePrices", "[]")
        try:
            outcome_prices = (
                json.loads(outcome_prices_str) if outcome_prices_str else []
            )
            # Convert to float
            outcome_prices = [float(p) for p in outcome_prices]
        except (json.JSONDecodeError, ValueError, TypeError):
            outcome_prices = []

        # Load each outcome
        for i, outcome_label in enumerate(outcomes):
            outcome_file = market_dir / f"outcome_{i}.parquet"

            if not outcome_file.exists():
                continue

            try:
                # Load price data
                df = pd.read_parquet(outcome_file)

                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                # Make timezone-aware if needed
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

                # Sample one row per day if requested
                if sample_per_day:
                    df[date_column] = df["timestamp"].dt.date
                    # Take the first observation of each day
                    df = df.groupby(date_column).first().reset_index()
                else:
                    df[date_column] = df["timestamp"].dt.date

                # Add slug and outcome
                df["slug"] = slug
                df["outcome"] = outcome_label

                # Add resolution price (0 or 1) if available
                if i < len(outcome_prices):
                    df["resolution_price"] = outcome_prices[i]
                else:
                    df["resolution_price"] = None

                # Calculate days_remaining
                if end_date is not None:
                    df["days_remaining"] = (
                        end_date - df["timestamp"]
                    ).dt.total_seconds() / (24 * 3600)
                else:
                    df["days_remaining"] = None

                # Add metadata fields
                df["question"] = metadata.get("question", "")
                df["end_date"] = end_date_str
                df["category"] = metadata.get("category", "")
                df["closed"] = metadata.get("full_info", {}).get("closed", False)

                # Add numeric metadata columns
                numeric_fields = {
                    "volume": "volume",
                    "liquidity": "liquidity",
                }

                for field_name, metadata_key in numeric_fields.items():
                    value = metadata.get(metadata_key)
                    if value is not None:
                        try:
                            df[field_name] = float(value)
                        except (ValueError, TypeError):
                            df[field_name] = None
                    else:
                        df[field_name] = None

                # Add additional numeric fields from full_info if available
                full_info = metadata.get("full_info", {})
                if full_info:
                    additional_numeric_fields = [
                        "volumeNum",
                        "liquidityNum",
                        "volume24hr",
                    ]

                    for field in additional_numeric_fields:
                        if field in full_info:
                            try:
                                df[field] = float(full_info[field])
                            except (ValueError, TypeError):
                                df[field] = None

                all_data.append(df)

            except Exception as e:
                print(f"  Warning: Failed to load outcome {i} for {slug}: {e}")
                continue

    if not all_data:
        print("Warning: No data loaded from any market")
        return pd.DataFrame()

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by slug, outcome, and timestamp
    combined_df = combined_df.sort_values(["slug", "outcome", "timestamp"]).reset_index(
        drop=True
    )

    # Add final_price: use resolution_price (0 or 1) if available, otherwise use last observed price
    def get_final_price(group):
        """Get the final price for a market outcome."""
        # Use resolution price if available (should be 0 or 1)
        resolution_price = group["resolution_price"].iloc[0]
        if pd.notna(resolution_price):
            return resolution_price
        else:
            # Fall back to last observed price
            return group["price"].iloc[-1]

    final_prices = combined_df.groupby(["slug", "outcome"]).apply(
        get_final_price, include_groups=False
    )
    combined_df["final_price"] = combined_df.apply(
        lambda row: final_prices.loc[(row["slug"], row["outcome"])], axis=1
    )

    # Add avg_daily_volume: total volume divided by days market was open
    # Calculate market duration for each slug/outcome
    def calculate_avg_daily_volume(group):
        """Calculate average daily volume for a market."""
        if len(group) == 0:
            return None

        # Get start and end timestamps
        start_time = group["timestamp"].min()
        end_time = group["timestamp"].max()

        # Calculate duration in days
        duration_days = (end_time - start_time).total_seconds() / (24 * 3600)

        # Avoid division by zero
        if duration_days == 0:
            duration_days = 1  # At least 1 day

        # Get total volume (should be same for all rows in group)
        total_volume = group["volume"].iloc[0]

        if pd.isna(total_volume) or total_volume == 0:
            return None

        return total_volume / duration_days

    # Calculate avg_daily_volume for each slug/outcome
    avg_daily_volumes = combined_df.groupby(["slug", "outcome"]).apply(
        calculate_avg_daily_volume, include_groups=False
    )
    combined_df["avg_daily_volume"] = combined_df.apply(
        lambda row: avg_daily_volumes.loc[(row["slug"], row["outcome"])], axis=1
    )

    print(f"Loaded {len(combined_df)} total rows")
    print(f"  Markets: {combined_df['slug'].nunique()}")
    print(f"  Outcomes: {combined_df['outcome'].nunique()}")
    print(
        f"  Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}"
    )

    return combined_df


if __name__ == "__main__":
    # Example usage
    print("Polymarket Data Downloader")
    print("=" * 50)

    # Example: Download one day of minute data for Fed rate hike market
    df = download_polymarket_prices_by_slug(
        market_slug="fed-rate-hike-in-2025",
        outcome_index=0,  # "Yes" outcome
        start_date=datetime(2024, 12, 1),
        end_date=datetime(2024, 12, 2),
        fidelity=1,  # 1 minute bars
    )

    print(f"\nData shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())
    print(f"\nPrice statistics:")
    print(df["price"].describe())
