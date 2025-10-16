# Market History Downloader

This document explains how to download complete price histories for Polymarket markets with all outcomes, organized in an efficient directory structure.

## Overview

The market history downloader provides a comprehensive solution for:
- Downloading complete price histories for any Polymarket market
- Handling all outcomes for each market
- Organizing data in market-specific folders with metadata
- Implementing rate limiting and retry logic to avoid server overload
- Caching data efficiently for later analysis

## Quick Start

```python
from market_history_downloader import download_market_complete_history

# Download a single market
result = download_market_complete_history(
    "will-donald-trump-be-inaugurated",
    fidelity=10  # 10-minute bars
)

print(f"Downloaded {result['outcomes_downloaded']} outcomes")
print(f"Total data points: {result['total_data_points']}")
```

## Directory Structure

Data is organized hierarchically:

```
data/polymarket/market_histories/
├── market-slug-1/
│   ├── metadata.json          # Market information
│   ├── summary.json            # Download summary
│   ├── outcome_0.parquet       # Price history for outcome 0
│   ├── outcome_1.parquet       # Price history for outcome 1
│   └── ...
├── market-slug-2/
│   └── ...
└── download_summary.json       # Overall download summary
```

### Example Directory

```
data/polymarket/market_histories/
├── will-donald-trump-be-inaugurated/
│   ├── metadata.json
│   ├── summary.json
│   ├── outcome_0.parquet       # "Yes" prices (11,452 data points)
│   └── outcome_1.parquet       # "No" prices (11,452 data points)
```

## Main Functions

### 1. download_market_complete_history()

Downloads complete price history for all outcomes of a single market.

```python
from market_history_downloader import download_market_complete_history

result = download_market_complete_history(
    market_slug="will-donald-trump-be-inaugurated",
    output_dir="data/polymarket/market_histories",
    fidelity=10,
    rate_limiter=None,  # Auto-created if not provided
    overwrite=False,
    max_retries=3
)
```

**Parameters:**
- `market_slug`: Market identifier (e.g., "will-donald-trump-be-inaugurated")
- `output_dir`: Base directory for data (default: "data/polymarket/market_histories")
- `fidelity`: Time resolution in minutes (default: 10)
- `rate_limiter`: Optional RateLimiter instance
- `overwrite`: If True, re-download even if cached
- `max_retries`: Retry attempts per outcome (default: 3)

**Returns:**
```python
{
    'slug': 'market-slug',
    'success': True,
    'outcomes_downloaded': 2,
    'total_data_points': 22904,
    'question': 'Market question text',
    'download_timestamp': '2025-10-15T...',
    'fidelity': 10,
    'date_range': {
        'start': '2024-11-01T...',
        'end': '2025-01-20T...',
        'days': 79
    }
}
```

### 2. download_market_list()

Downloads multiple markets in sequence.

```python
from market_history_downloader import download_market_list

market_slugs = [
    'will-donald-trump-be-inaugurated',
    'will-zelenskyy-wear-a-suit-before-july',
    'tiktok-banned-in-the-us-before-may-2025'
]

summary = download_market_list(
    market_slugs=market_slugs,
    output_dir="data/polymarket/market_histories",
    fidelity=10,
    overwrite=False
)

print(f"Downloaded {summary['successful']}/{summary['total_markets']} markets")
print(f"Total data points: {summary['total_data_points']:,}")
```

**Returns:**
```python
{
    'total_markets': 3,
    'successful': 3,
    'failed': 0,
    'total_outcomes': 6,
    'total_data_points': 45000,
    'elapsed_time': 120.5,
    'results': [...]  # Individual results
}
```

### 3. load_market_data()

Load downloaded data from disk.

```python
from market_history_downloader import load_market_data

data = load_market_data('will-donald-trump-be-inaugurated')

print(data['metadata']['question'])
print(f"Outcomes: {data['outcome_names']}")

# Access price data
for name, df in zip(data['outcome_names'], data['outcomes']):
    print(f"\n{name}:")
    print(f"  Data points: {len(df)}")
    print(f"  Price range: {df['price'].min():.3f} - {df['price'].max():.3f}")
    print(df.head())
```

**Returns:**
```python
{
    'metadata': {...},        # Full market metadata
    'summary': {...},          # Download summary
    'outcomes': [df1, df2],   # List of DataFrames
    'outcome_names': ['Yes', 'No']
}
```

## Rate Limiting

The downloader includes built-in rate limiting to avoid overloading Polymarket servers.

### RateLimiter Class

```python
from market_history_downloader import RateLimiter

# Create rate limiter
limiter = RateLimiter(
    min_delay=0.5,   # 0.5 seconds between requests
    max_delay=60.0   # Max 60 seconds on exponential backoff
)

# Use with downloads
result = download_market_complete_history(
    "market-slug",
    rate_limiter=limiter
)
```

**How it works:**
- Enforces minimum delay between API requests (default: 0.5s)
- Implements exponential backoff on errors
- Resets error count on successful requests
- Maximum backoff capped at `max_delay` (default: 60s)

**Exponential Backoff:**
- Error 1: 0.5s delay
- Error 2: 1.0s delay
- Error 3: 2.0s delay
- Error 4: 4.0s delay
- Error N: min(0.5 * 2^N, 60.0)s delay

## Selected Markets Download

The repository includes pre-selected markets for analysis:

### By Total Volume
```bash
python download_selected_markets.py
```

Downloads 30 markets:
- 10 high volume (>$76M)
- 10 medium volume (~$11k)
- 10 low volume (<$1)

### By Weekly Volume
```bash
python download_selected_markets_volume1wk.py
```

Downloads 30 markets by `volume1wk`:
- 10 high weekly volume (>$19M)
- 10 medium weekly volume (~$11k)
- 10 low weekly volume (<$1)

## Data Files

### metadata.json

Complete market information:
```json
{
  "slug": "will-donald-trump-be-inaugurated",
  "question": "Will Donald Trump be inaugurated?",
  "outcomes": ["Yes", "No"],
  "token_ids": ["457...", "819..."],
  "start_date": "2024-11-01T20:59:58.040376+00:00",
  "end_date": "2025-01-20T12:00:00+00:00",
  "volume": "400409526.886835",
  "liquidity": 0,
  "category": "Unknown",
  "download_timestamp": "2025-10-15T19:11:27.562613",
  "fidelity": 10,
  "full_info": {...}  // Complete API response
}
```

### summary.json

Download summary:
```json
{
  "slug": "will-donald-trump-be-inaugurated",
  "question": "Will Donald Trump be inaugurated?",
  "success": true,
  "outcomes_downloaded": 2,
  "total_outcomes": 2,
  "total_data_points": 22904,
  "download_timestamp": "2025-10-15T19:11:27.562613",
  "fidelity": 10,
  "date_range": {
    "start": "2024-11-01T20:59:58.040376+00:00",
    "end": "2025-01-20T12:00:00+00:00",
    "days": 79
  }
}
```

### outcome_N.parquet

Price history DataFrame with columns:
- `timestamp`: pandas datetime64 with timezone
- `price`: float (0-1 probability)
- `unix_timestamp`: int (Unix timestamp in seconds)

## Example Workflows

### 1. Download and Analyze Single Market

```python
from market_history_downloader import (
    download_market_complete_history,
    load_market_data
)
import pandas as pd

# Download
result = download_market_complete_history(
    "will-donald-trump-be-inaugurated",
    fidelity=60  # Hourly data
)

# Load
data = load_market_data("will-donald-trump-be-inaugurated")

# Analyze
yes_prices = data['outcomes'][0]
print(f"Trump inauguration probability:")
print(f"  Start: {yes_prices.iloc[0]['price']:.3f}")
print(f"  End: {yes_prices.iloc[-1]['price']:.3f}")
print(f"  Max: {yes_prices['price'].max():.3f}")
print(f"  Min: {yes_prices['price'].min():.3f}")

# Plot
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=yes_prices['timestamp'],
    y=yes_prices['price'],
    name='Yes',
    line=dict(color='green')
))
fig.update_layout(
    title="Trump Inauguration Probability Over Time",
    xaxis_title="Date",
    yaxis_title="Probability",
    yaxis=dict(range=[0, 1])
)
fig.show()
```

### 2. Batch Download Multiple Markets

```python
from polymarket_data import get_all_closed_markets
from market_history_downloader import download_market_list

# Get all closed markets
all_markets = get_all_closed_markets()

# Filter for high-volume political markets
politics = all_markets[
    (all_markets['category'].str.contains('affairs', na=False)) &
    (all_markets['volume_num'] > 1_000_000)
].sort_values('volume_num', ascending=False)

# Download top 10
slugs = politics.head(10)['slug'].tolist()

summary = download_market_list(
    market_slugs=slugs,
    fidelity=10
)

print(f"Success rate: {summary['successful']}/{summary['total_markets']}")
```

### 3. Compare Multiple Outcomes

```python
from market_history_downloader import load_market_data
import plotly.graph_objects as go

data = load_market_data("will-donald-trump-be-inaugurated")

fig = go.Figure()

for idx, (name, df) in enumerate(zip(data['outcome_names'], data['outcomes'])):
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price'],
        name=name,
        line=dict(width=2)
    ))

fig.update_layout(
    title=data['metadata']['question'],
    xaxis_title="Date",
    yaxis_title="Probability",
    height=600
)
fig.show()
```

### 4. Statistical Analysis

```python
from market_history_downloader import load_market_data
import pandas as pd

data = load_market_data("will-donald-trump-be-inaugurated")

for name, df in zip(data['outcome_names'], data['outcomes']):
    print(f"\n{name} Statistics:")
    print(df['price'].describe())

    # Daily volatility
    df['returns'] = df['price'].pct_change()
    daily_vol = df['returns'].std() * (144 ** 0.5)  # 144 10-min periods per day
    print(f"Daily volatility: {daily_vol:.4f}")

    # Price changes
    start_price = df.iloc[0]['price']
    end_price = df.iloc[-1]['price']
    change = end_price - start_price
    print(f"Total change: {change:+.4f} ({change/start_price*100:+.2f}%)")
```

## Performance

### Download Speeds
- ~2,000 data points per 14-day chunk
- ~0.5-1 second per API request (with rate limiting)
- Example: 79-day market = 6 chunks ≈ 5-10 seconds per outcome

### Caching
- Downloaded data cached in Parquet format
- Subsequent loads: <0.1 seconds
- No re-downloading unless `overwrite=True`

### Storage
- 10-minute fidelity: ~2,000 points per 14 days
- Typical binary market (2 outcomes, 30 days): ~600 KB
- 30 markets with 60 outcomes total: ~15 MB

## Error Handling

### Automatic Retry
Functions automatically retry on failure:
- Network errors: 3 attempts with exponential backoff
- API timeouts: Handled gracefully
- Rate limit errors: Automatic backoff

### Handling Failures

```python
result = download_market_complete_history("market-slug")

if not result['success']:
    print(f"Failed: {result.get('error', 'Unknown error')}")
else:
    partial = result['outcomes_downloaded'] < result.get('total_outcomes', 0)
    if partial:
        print(f"Partial success: {result['outcomes_downloaded']} outcomes")
```

### Common Issues

**Issue: "Market with slug '...' not found"**
- Solution: Verify slug is correct using `get_all_closed_markets()`

**Issue: Download very slow**
- Solution: Markets with long date ranges are automatically chunked. This is normal.

**Issue: "Invalid comparison between dtype..."**
- Solution: Update `polymarket_data.py` (fixed in latest version)

## Best Practices

1. **Use Rate Limiting**: Always use `RateLimiter` for batch downloads
2. **Check Cache First**: Set `overwrite=False` to avoid re-downloading
3. **Handle Errors**: Check `result['success']` before proceeding
4. **Monitor Progress**: Watch `download_log_fixed.txt` for batch operations
5. **Start Small**: Test with `max_markets` parameter first
6. **Fidelity Choice**: Use 10-60 minute bars for analysis (1-minute rarely needed)

## Troubleshooting

### Check Download Progress
```bash
tail -f data/polymarket/download_log_fixed.txt
```

### Check Downloaded Markets
```bash
ls -la data/polymarket/market_histories/
```

### Verify Data Quality
```python
from market_history_downloader import load_market_data

data = load_market_data("market-slug")

# Check for missing data
for name, df in zip(data['outcome_names'], data['outcomes']):
    print(f"{name}: {len(df)} points, {df['price'].isna().sum()} missing")
```

## See Also

- **polymarket_data.py**: Core data download functions
- **POLYMARKET_DATA_README.md**: API documentation
- **CLOSED_MARKETS_README.md**: Market discovery and filtering
- **download_selected_markets.py**: Example batch download script
