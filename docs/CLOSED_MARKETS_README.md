# Polymarket Closed Markets Metadata

This document describes the `get_all_closed_markets()` function for fetching metadata on all closed Polymarket markets.

## Overview

The `get_all_closed_markets()` function provides comprehensive metadata for all closed Polymarket prediction markets. This data is essential for analyzing historical market performance, selecting markets for deeper analysis, and understanding the Polymarket ecosystem.

## Quick Start

```python
from polymarket_data import get_all_closed_markets

# Fetch all closed markets (uses cache if available)
df = get_all_closed_markets()

print(f"Loaded {len(df)} closed markets")
print(f"Total volume: ${df['volume_num'].sum():,.2f}")

# View top markets by volume
top_markets = df.nlargest(10, 'volume_num')[['question', 'volume_num', 'category']]
print(top_markets)
```

## Function Signature

```python
def get_all_closed_markets(
    cache_dir: str = "data/polymarket",
    overwrite: bool = False,
    max_markets: Optional[int] = None
) -> pd.DataFrame
```

### Parameters

- **cache_dir** (str, default="data/polymarket"): Directory to store cached data
- **overwrite** (bool, default=False): If True, re-fetch from API even if cache exists
- **max_markets** (Optional[int], default=None): Maximum number of markets to fetch (for testing)

### Returns

- **pd.DataFrame**: DataFrame with comprehensive metadata for all closed markets

## Dataset Statistics

As of the latest fetch:

- **Total Markets**: 58,980 closed markets
- **Total Volume**: $11.4+ billion
- **Date Range**: 2018-06-13 to present
- **Categories**: Sports, US-current-affairs, Crypto, Pop-Culture, and many more
- **File Size**: ~42 MB (Parquet compressed)

## DataFrame Structure

### Key Columns

The DataFrame contains 138 columns with comprehensive market data:

#### Essential Fields
- **slug** (str): Unique market identifier URL slug
- **question** (str): The market question
- **outcomes** (list): Possible outcomes for the market
- **outcomePrices** (list): Final outcome prices
- **volume_num** (float): Total trading volume in USD
- **liquidity_num** (float): Current liquidity in USD
- **category** (str): Market category

#### Dates
- **end_date_parsed** (datetime): When the market ended
- **created_at_parsed** (datetime): When the market was created
- **closed_time_parsed** (datetime): When the market was closed/resolved

#### Metadata
- **id** (str): Internal market ID
- **conditionId** (str): Condition identifier
- **marketType** (str): Type of market (binary, categorical, etc.)
- **closed** (bool): Whether market is closed (always True in this dataset)
- **clobTokenIds** (list): Token IDs for CLOB trading

#### Volume Breakdowns
- **volume24hr**, **volume1wk**, **volume1mo**, **volume1yr**: Volume by time period
- **volume1wkAmm**, **volume1wkClob**: Volume by trading venue

#### Price Data
- **lastTradePrice** (float): Last trade price before close
- **bestBid**, **bestAsk** (float): Final bid/ask prices
- **oneDayPriceChange**, **oneWeekPriceChange**, etc.: Historical price movements

## Usage Examples

### 1. Basic Loading

```python
from polymarket_data import get_all_closed_markets

# First run - downloads all markets (takes ~2 minutes)
df = get_all_closed_markets()

# Subsequent runs - loads from cache instantly
df = get_all_closed_markets()
```

### 2. Filtering High-Volume Markets

```python
# Markets with >$1M volume
high_volume = df[df['volume_num'] > 1_000_000]

print(f"Found {len(high_volume)} markets with >$1M volume")
print(high_volume[['question', 'volume_num', 'category']].head())
```

### 3. Filtering by Category

```python
# Political markets
politics = df[df['category'].str.lower().str.contains('politics', na=False)]

# Sports markets
sports = df[df['category'] == 'Sports']

# Crypto markets
crypto = df[df['category'] == 'Crypto']
```

### 4. Filtering by Date

```python
import pandas as pd

# Markets from last year
recent_cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=365)
recent = df[df['end_date_parsed'] > recent_cutoff]

# Markets from specific date range
start = pd.Timestamp('2024-01-01', tz='UTC')
end = pd.Timestamp('2024-12-31', tz='UTC')
year_2024 = df[
    (df['end_date_parsed'] >= start) &
    (df['end_date_parsed'] <= end)
]
```

### 5. Combining Filters

```python
# High-volume recent political markets
filtered = df[
    (df['volume_num'] > 500_000) &
    (df['category'].str.contains('current-affairs', na=False)) &
    (df['end_date_parsed'] > pd.Timestamp('2024-01-01', tz='UTC'))
]

# Sort by volume
filtered_sorted = filtered.sort_values('volume_num', ascending=False)
print(filtered_sorted[['question', 'volume_num']].head(10))
```

### 6. Exporting Filtered Data

```python
# Export high-volume markets to CSV
high_volume = df[df['volume_num'] > 1_000_000]

# Select key columns
export_cols = [
    'slug', 'question', 'volume_num', 'liquidity_num',
    'category', 'end_date_parsed', 'outcomes', 'outcomePrices'
]

high_volume[export_cols].to_csv('high_volume_markets.csv', index=False)
```

### 7. Analyzing by Category

```python
# Count markets by category
category_counts = df['category'].value_counts()
print("Top 10 categories:")
print(category_counts.head(10))

# Total volume by category
volume_by_category = df.groupby('category')['volume_num'].sum().sort_values(ascending=False)
print("\nTop categories by volume:")
print(volume_by_category.head(10))
```

### 8. Finding Specific Markets

```python
# Search by keyword in question
search_term = "Trump"
results = df[df['question'].str.contains(search_term, case=False, na=False)]

print(f"Found {len(results)} markets containing '{search_term}'")
print(results[['question', 'volume_num']].head())

# Get token IDs for downloading price data
for idx, row in results.head(5).iterrows():
    print(f"\nMarket: {row['question'][:60]}...")
    print(f"  Slug: {row['slug']}")
    print(f"  Token IDs: {row['clobTokenIds']}")
    print(f"  Volume: ${row['volume_num']:,.2f}")
```

## Integration with Price Data

Use the metadata to select markets, then download their price history:

```python
from polymarket_data import get_all_closed_markets, download_polymarket_prices
from datetime import datetime, timedelta

# 1. Get all closed markets
markets = get_all_closed_markets()

# 2. Filter for high-volume political markets
interesting_markets = markets[
    (markets['volume_num'] > 5_000_000) &
    (markets['category'] == 'US-current-affairs')
]

# 3. Download price data for the top market
top_market = interesting_markets.nlargest(1, 'volume_num').iloc[0]
token_id = top_market['clobTokenIds'][0]

# Download one month before market close
end_date = top_market['end_date_parsed']
start_date = end_date - timedelta(days=30)

prices = download_polymarket_prices(
    token_id=token_id,
    start_date=start_date.to_pydatetime(),
    end_date=end_date.to_pydatetime(),
    fidelity=60  # Hourly data
)

print(f"Downloaded {len(prices)} price points for: {top_market['question']}")
```

## Caching Details

### Cache Location
- Default: `data/polymarket/closed_markets_metadata.parquet`
- Format: Parquet (compressed, efficient)
- Size: ~42 MB for full dataset

### Cache Behavior
- **First run**: Downloads all markets from API (~2 minutes for 59k markets)
- **Subsequent runs**: Loads from cache instantly (<0.1 seconds)
- **Overwrite**: Use `overwrite=True` to force fresh download

```python
# Load from cache
df = get_all_closed_markets(overwrite=False)  # Default

# Force fresh download
df = get_all_closed_markets(overwrite=True)
```

### Cache Management

```python
from pathlib import Path
import pandas as pd

# Check cache status
cache_file = Path("data/polymarket/closed_markets_metadata.parquet")

if cache_file.exists():
    print(f"Cache exists: {cache_file}")
    print(f"Size: {cache_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Modified: {datetime.fromtimestamp(cache_file.stat().st_mtime)}")

    # Quick peek at cached data
    df = pd.read_parquet(cache_file)
    print(f"Markets: {len(df)}")
else:
    print("No cache found - will download on first use")
```

## Performance

### Benchmarks
- **Initial Download**: ~2 minutes for 59k markets
- **Cache Load**: <0.1 seconds
- **Pagination**: API returns 500 markets per page, ~118 pages total
- **Memory Usage**: ~200 MB for full DataFrame

### Optimization Tips
1. **Use cache**: Don't set `overwrite=True` unless necessary
2. **Filter early**: Reduce DataFrame size with filters before processing
3. **Select columns**: Use `df[columns]` to work with subset
4. **Test with max_markets**: Use `max_markets=100` for testing code

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/test_closed_markets.py -v

# Run specific test
pytest tests/test_closed_markets.py::TestClosedMarketsMetadata::test_fetch_limited_markets -v
```

Run the example script:

```bash
python closed_markets_example.py
```

## API Details

### Endpoint
- **URL**: `https://gamma-api.polymarket.com/markets`
- **Method**: GET
- **Parameters**:
  - `closed=true`: Only closed markets
  - `limit=1000`: Markets per page (max 1000, API returns 500)
  - `offset=N`: Pagination offset

### Rate Limiting
- No explicit rate limit documented
- Function includes 1-second delay between failed requests
- Retries up to 3 times on timeout

### Response Format
Returns JSON array of market objects with 100+ fields per market.

## Common Use Cases

### 1. Market Discovery
Find interesting markets to analyze based on volume, category, or recency.

### 2. Research & Analysis
Study historical market performance, accuracy, and trading patterns.

### 3. Model Training
Use closed markets with known outcomes as training data for prediction models.

### 4. Categorization
Analyze which types of markets generate the most trading activity.

### 5. Time Series Analysis
Track how certain market types evolve over time.

### 6. Outcome Analysis
Study final prices vs actual outcomes to assess market efficiency.

## Troubleshooting

### Issue: First download is slow
**Solution**: This is expected. 59k markets take ~2 minutes to download. Subsequent loads use cache and are instant.

### Issue: Cache seems outdated
**Solution**: Use `overwrite=True` to force fresh download:
```python
df = get_all_closed_markets(overwrite=True)
```

### Issue: Memory errors with full dataset
**Solution**: Filter early to reduce DataFrame size:
```python
# Load and immediately filter
df = get_all_closed_markets()
df = df[df['volume_num'] > 100_000]  # Keep only high-volume
```

### Issue: Missing or NaT dates
**Solution**: Some old markets have missing metadata. Filter these out:
```python
df = df[df['end_date_parsed'].notna()]
```

## Related Functions

- **get_market_info(slug)**: Get metadata for a specific market
- **download_polymarket_prices(token_id, ...)**: Download price history
- **download_polymarket_prices_by_slug(slug, ...)**: Download prices by slug
- **get_event_markets(event_slug)**: Get markets within an event

## Example Workflow

Complete workflow for analyzing a specific market type:

```python
from polymarket_data import get_all_closed_markets, download_polymarket_prices
import pandas as pd
from datetime import timedelta

# 1. Load all closed markets
print("Loading closed markets...")
markets = get_all_closed_markets()

# 2. Filter for high-volume crypto markets
crypto_markets = markets[
    (markets['category'] == 'Crypto') &
    (markets['volume_num'] > 100_000)
].sort_values('volume_num', ascending=False)

print(f"Found {len(crypto_markets)} high-volume crypto markets")

# 3. Select top 5 markets
top_5 = crypto_markets.head(5)

# 4. Download price data for each
for idx, market in top_5.iterrows():
    print(f"\nAnalyzing: {market['question']}")

    # Get date range (30 days before close)
    end_date = market['end_date_parsed'].to_pydatetime()
    start_date = end_date - timedelta(days=30)

    # Download hourly prices
    token_id = market['clobTokenIds'][0]
    prices = download_polymarket_prices(
        token_id=token_id,
        start_date=start_date,
        end_date=end_date,
        fidelity=60
    )

    # Analyze
    print(f"  Price range: {prices['price'].min():.3f} - {prices['price'].max():.3f}")
    print(f"  Final price: {prices['price'].iloc[-1]:.3f}")
    print(f"  Volume: ${market['volume_num']:,.0f}")
```

## See Also

- **POLYMARKET_DATA_README.md**: Complete Polymarket data documentation
- **tests/test_closed_markets.py**: Test suite with examples
- **closed_markets_example.py**: Runnable example script
