# Polymarket Data Download Module

This module provides functionality to download and cache historical price data from Polymarket's CLOB (Central Limit Order Book) API.

## Overview

The `polymarket_data.py` module allows you to:
- Download minute-level (or any resolution) price data for Polymarket prediction markets
- Automatically cache data to disk to avoid redundant API calls
- Look up market information by slug
- Download data for specific date ranges or entire months

## Installation

Install the required dependencies:

```bash
pip install pandas requests pyarrow
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Example 1: Download one day of minute bar data

```python
from datetime import datetime
from polymarket_data import download_polymarket_prices

# Download minute-level data for a specific token
df = download_polymarket_prices(
    token_id="60487116984468020978247225474488676749601001829886755968952521846780452448915",
    start_date=datetime(2024, 10, 28),
    end_date=datetime(2024, 10, 29),
    fidelity=1,  # 1 minute bars
    data_dir="data/polymarket",
    overwrite=False
)

print(df.head())
```

### Example 2: Download one month of data

```python
from polymarket_data import download_month_of_data

# Download October 2024 data
df = download_month_of_data(
    token_id="60487116984468020978247225474488676749601001829886755968952521846780452448915",
    year=2024,
    month=10,
    fidelity=60  # Hourly bars
)

print(f"Downloaded {len(df)} data points")
```

### Example 3: Download by market slug

```python
from polymarket_data import download_polymarket_prices_by_slug
from datetime import datetime

# Download data using market slug instead of token ID
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    outcome_index=0,  # 0 = first outcome (e.g., "Yes"), 1 = second outcome (e.g., "No")
    start_date=datetime(2024, 12, 1),
    end_date=datetime(2024, 12, 2),
    fidelity=1
)
```

## API Reference

### Core Functions

#### `download_polymarket_prices()`

Download historical price data for a Polymarket token.

**Parameters:**
- `token_id` (str): The CLOB token ID (large integer as string)
- `start_date` (datetime): Start date for historical data
- `end_date` (datetime): End date for historical data
- `fidelity` (int): Data resolution in minutes (default: 1 for minute bars)
- `data_dir` (str): Directory to store cached data (default: "data/polymarket")
- `overwrite` (bool): If True, download even if cached data exists (default: False)

**Returns:**
- pandas DataFrame with columns:
  - `timestamp` (datetime): Timestamp of the data point
  - `price` (float): Price value (probability between 0 and 1)
  - `unix_timestamp` (int): Unix timestamp in seconds

**Raises:**
- `ValueError`: If invalid parameters provided
- `requests.RequestException`: If API request fails

---

#### `download_month_of_data()`

Download one month of historical price data (convenience wrapper).

**Parameters:**
- `token_id` (str): The CLOB token ID
- `year` (int): Year (e.g., 2024)
- `month` (int): Month (1-12)
- `fidelity` (int): Data resolution in minutes (default: 1)
- `data_dir` (str): Directory to store cached data (default: "data/polymarket")
- `overwrite` (bool): If True, download even if cached data exists (default: False)

**Returns:**
- pandas DataFrame (same format as `download_polymarket_prices()`)

---

#### `download_polymarket_prices_by_slug()`

Download historical price data by market slug (looks up token ID automatically).

**Parameters:**
- `market_slug` (str): The market slug (e.g., "fed-rate-hike-in-2025")
- `outcome_index` (int): Index of the outcome (0 for first, 1 for second, etc.)
- `start_date` (datetime): Start date for historical data
- `end_date` (datetime): End date for historical data
- `fidelity` (int): Data resolution in minutes (default: 1)
- `data_dir` (str): Directory to store cached data (default: "data/polymarket")
- `overwrite` (bool): If True, download even if cached data exists (default: False)

**Returns:**
- pandas DataFrame (same format as `download_polymarket_prices()`)

---

#### `get_market_info()`

Get market information from Polymarket's Gamma API.

**Parameters:**
- `market_slug` (str): The market slug
- `max_pages` (int): Maximum number of pages to search (default: 10, i.e., 10,000 markets)

**Returns:**
- Dictionary containing market information including:
  - `question`: Market question text
  - `slug`: Market slug
  - `clobTokenIds`: JSON string of token IDs for each outcome
  - `outcomes`: JSON string of outcome names
  - And many other fields...

**Raises:**
- `ValueError`: If market not found
- `requests.RequestException`: If API request fails

## Caching Mechanism

The module implements an intelligent caching system:

1. **Automatic Caching**: Downloaded data is automatically saved to `data/polymarket/` as Parquet files
2. **Cache Naming**: Files are named based on parameters: `token_{id}_{start}_{end}_fid{fidelity}.parquet`
3. **Cache Loading**: If a file already exists, it's loaded from disk instead of making an API call
4. **Overwrite Flag**: Set `overwrite=True` to force re-download even if cache exists

### Example: Using the cache

```python
# First call - downloads from API and caches
df1 = download_polymarket_prices(...)  # API call made

# Second call - loads from cache (no API call)
df2 = download_polymarket_prices(...)  # Loaded from disk

# Force re-download
df3 = download_polymarket_prices(..., overwrite=True)  # API call made
```

## Finding Token IDs

Token IDs are large integers that identify specific outcomes in Polymarket markets. There are several ways to find them:

### Method 1: Use the Gamma API directly

```python
from polymarket_data import get_market_info
import json

market_info = get_market_info("fed-rate-hike-in-2025")
token_ids = json.loads(market_info["clobTokenIds"])
outcomes = json.loads(market_info["outcomes"])

for i, (token_id, outcome) in enumerate(zip(token_ids, outcomes)):
    print(f"Outcome {i} ({outcome}): {token_id}")
```

### Method 2: Browse the Gamma API

Visit the markets endpoint in your browser:
```
https://gamma-api.polymarket.com/markets?limit=10
```

Look for the `clobTokenIds` field in the market data.

### Method 3: Use the slug-based function

Just use `download_polymarket_prices_by_slug()` which looks up token IDs automatically.

## Data Format

Downloaded data is returned as a pandas DataFrame with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime64 | Human-readable timestamp (UTC) |
| `price` | float64 | Price value (probability between 0 and 1) |
| `unix_timestamp` | int64 | Unix timestamp in seconds |

Example:
```
                 timestamp  price  unix_timestamp
0  2024-10-28 00:00:00       0.65    1698451200
1  2024-10-28 00:01:00       0.66    1698451260
2  2024-10-28 00:02:00       0.65    1698451320
```

## API Documentation

This module uses two Polymarket APIs:

1. **CLOB API**: For historical price data
   - Base URL: `https://clob.polymarket.com`
   - Endpoint: `/prices-history`
   - Documentation: https://docs.polymarket.com/developers/CLOB/timeseries

2. **Gamma Markets API**: For market metadata
   - Base URL: `https://gamma-api.polymarket.com`
   - Endpoint: `/markets`
   - Documentation: https://docs.polymarket.com/developers/gamma-markets-api/overview

## Testing

Run the unit tests:

```bash
# Run all tests
pytest tests/test_polymarket_download.py -v

# Run specific test
pytest tests/test_polymarket_download.py::TestPolymarketDataDownload::test_download_one_day_minute_data -v -s
```

The test suite includes:
- ✓ Download one day of minute bar data
- ✓ Caching mechanism validation
- ✓ Overwrite flag functionality
- ✓ Invalid parameter rejection
- ✓ Error handling

## Common Use Cases

### Case 1: Analyzing price movement around an event

```python
from datetime import datetime, timedelta

# Download data for one week before election day
election_day = datetime(2024, 11, 5)
week_before = election_day - timedelta(days=7)

df = download_polymarket_prices(
    token_id="YOUR_TOKEN_ID",
    start_date=week_before,
    end_date=election_day,
    fidelity=60  # Hourly data
)

# Analyze price volatility
df['price_change'] = df['price'].pct_change()
print(f"Max hourly change: {df['price_change'].abs().max():.2%}")
```

### Case 2: Building a time series dataset

```python
import pandas as pd

# Download multiple months of data
dfs = []
for month in range(1, 13):
    df = download_month_of_data(
        token_id="YOUR_TOKEN_ID",
        year=2024,
        month=month,
        fidelity=1440  # Daily bars
    )
    dfs.append(df)

# Combine into single dataset
full_year = pd.concat(dfs, ignore_index=True)
print(f"Total data points: {len(full_year)}")
```

### Case 3: Comparing multiple outcomes

```python
# Download data for both outcomes of a binary market
market_slug = "fed-rate-hike-in-2025"
start = datetime(2024, 12, 1)
end = datetime(2024, 12, 31)

yes_df = download_polymarket_prices_by_slug(
    market_slug=market_slug,
    outcome_index=0,  # "Yes"
    start_date=start,
    end_date=end
)

no_df = download_polymarket_prices_by_slug(
    market_slug=market_slug,
    outcome_index=1,  # "No"
    start_date=start,
    end_date=end
)

# Plot both outcomes
import matplotlib.pyplot as plt
plt.plot(yes_df['timestamp'], yes_df['price'], label='Yes')
plt.plot(no_df['timestamp'], no_df['price'], label='No')
plt.legend()
plt.show()
```

## Troubleshooting

### Problem: "Market with slug 'X' not found"

**Solution**: The market might be older and beyond the default pagination limit. Try increasing `max_pages`:

```python
market_info = get_market_info("your-market-slug", max_pages=20)
```

### Problem: "400 Bad Request" error

**Possible causes:**
1. Invalid token ID
2. Date range in the future or too far in the past
3. Very large fidelity value (try smaller values like 1, 60, or 1440)

### Problem: Empty DataFrame returned

**Explanation**: Some markets don't have data for all date ranges. This is expected behavior if:
- The market didn't exist during that period
- There was no trading activity during that period

### Problem: Slow performance

**Solutions:**
1. Use larger fidelity values (e.g., 60 for hourly instead of 1 for minute-level)
2. Download smaller date ranges
3. Make sure caching is enabled (`overwrite=False`)

## Performance Tips

1. **Use appropriate fidelity**: Minute data (fidelity=1) creates large files. Use hourly (60) or daily (1440) for long time periods.

2. **Leverage caching**: The first download is slow, but subsequent loads are instant.

3. **Download in chunks**: For very long time periods, download month by month:
   ```python
   for month in range(1, 13):
       df = download_month_of_data(token_id, 2024, month)
   ```

4. **Store as Parquet**: The module uses Parquet format which is much faster and smaller than CSV.

## Integration with Other Modules

This module integrates well with the other forecasting tools in this repository:

```python
# Example: Analyze buzz scores for markets with high trading activity
from polymarket_data import download_polymarket_prices
from buzz import get_buzz_score_openrouter
from datetime import datetime, timedelta

# Download recent price data
df = download_polymarket_prices(
    token_id="YOUR_TOKEN_ID",
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    fidelity=60
)

# Calculate volatility
volatility = df['price'].std()

# Get buzz score for the market
buzz = get_buzz_score_openrouter("Market question here")

print(f"Volatility: {volatility:.4f}, Buzz: {buzz:.4f}")
```

## Future Enhancements

Potential improvements for this module:
- Support for WebSocket streaming data
- Integration with on-chain data via Polygon RPC
- Batch download for multiple markets
- Advanced caching with TTL (time-to-live)
- Export to different formats (CSV, JSON, etc.)
- Support for order book depth data

## Contributing

When contributing to this module:
1. Add type hints to all functions
2. Write unit tests for new features
3. Update this README with examples
4. Follow the existing code style
5. Ensure tests pass: `pytest tests/test_polymarket_download.py`

## License

This module is part of the eterniscollab forecasting toolkit.

## Support

For issues or questions:
1. Check this README first
2. Review the unit tests for examples
3. Check Polymarket's official API documentation
4. Open an issue in the repository

## Acknowledgments

- Polymarket API documentation: https://docs.polymarket.com/
- Built using pandas, requests, and pyarrow
- Inspired by the need for historical data analysis of prediction markets
