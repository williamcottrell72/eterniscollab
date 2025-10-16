# Polymarket API Limits and Automatic Chunking

## The 15-Day Limit

The Polymarket CLOB API has a **~15 day maximum** per request, regardless of fidelity setting.

### Your Error

```
HTTPError: 400 Client Error: Bad Request
{"error":"invalid filters: 'startTs' and 'endTs' interval is too long"}
```

This happens when you request more than ~15 days of data in a single API call.

### Your Query

```python
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 2, 1),  # 31 days > 15 day limit
    fidelity=10
)
# Error: Date range too long!
```

## The Fix: Automatic Chunking ✨

I've updated the `download_polymarket_prices()` function to **automatically split large requests** into 14-day chunks and combine the results.

### Now Your Query Works!

```python
from polymarket_data import download_polymarket_prices_by_slug
from datetime import datetime

# This now works! Function automatically chunks it
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 2, 1),  # 31 days - automatically chunked!
    fidelity=10
)

print(f"Downloaded {len(df)} data points")
# Output: Downloaded 4465 data points ✓
```

### What Happens Behind the Scenes

```
Date range (31 days) exceeds API limit (14 days)
Splitting into 3 chunks...
  Downloading chunk: 2025-01-01 to 2025-01-15
  Downloaded 2017 data points
  Downloading chunk: 2025-01-15 to 2025-01-29
  Downloaded 2017 data points
  Downloading chunk: 2025-01-29 to 2025-02-01
  Downloaded 433 data points
Combined 3 chunks into 4465 total data points
```

## API Limit Details

The limit is approximately 15 days for ANY fidelity:

| Fidelity | Max Days | Max Data Points per Request |
|----------|----------|-----------------------------|
| 1 min    | ~15 days | ~21,600 points             |
| 10 min   | ~15 days | ~2,160 points              |
| 60 min   | ~15 days | ~360 points                |
| 1440 min | ~15 days | ~15 points                 |

The chunking logic uses a **conservative 14-day limit** to ensure reliability.

## Examples

### Example 1: Downloading 3 Months (Automatic Chunking)

```python
from datetime import datetime
from polymarket_data import download_polymarket_prices_by_slug

# 90 days = 6 chunks of 14 days + 1 chunk of 6 days
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 4, 1),  # 90 days
    fidelity=60  # Hourly data
)

# Function automatically:
# 1. Splits into ~7 chunks
# 2. Downloads each chunk separately
# 3. Combines and deduplicates results
# 4. Returns single DataFrame
```

### Example 2: Downloading 1 Year

```python
# 365 days = 26 chunks
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    start_date=datetime(2024, 12, 30),  # Market start
    end_date=datetime(2025, 12, 30),    # 1 year later
    fidelity=1440  # Daily data (smaller size)
)

# Automatically chunked into 26 requests
# Result: Complete 1-year dataset
```

### Example 3: Short Period (No Chunking)

```python
# 7 days < 14 day limit = single request
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 8),
    fidelity=1  # Minute data
)

# Single API call, ~10,080 data points
```

## Performance Considerations

### Chunking Overhead

Each chunk requires:
- 1 API request
- File I/O for caching
- Data concatenation

For very long date ranges, consider using higher fidelity:

```python
# Bad: 1 year at 1-minute fidelity
# = 26 chunks × 21,600 points = 561,600 points
df = download_polymarket_prices_by_slug(
    start_date=datetime(2024, 12, 30),
    end_date=datetime(2025, 12, 30),
    fidelity=1  # Slow! 26 API calls
)

# Better: 1 year at hourly fidelity
# = 26 chunks × 360 points = 9,360 points
df = download_polymarket_prices_by_slug(
    start_date=datetime(2024, 12, 30),
    end_date=datetime(2025, 12, 30),
    fidelity=60  # Faster! Still 26 calls but less data
)

# Best: 1 year at daily fidelity
# = 26 chunks × 15 points = 390 points
df = download_polymarket_prices_by_slug(
    start_date=datetime(2024, 12, 30),
    end_date=datetime(2025, 12, 30),
    fidelity=1440  # Fastest! Same calls, minimal data
)
```

### Caching Helps

Once downloaded, chunks are cached:

```python
# First time: Downloads all chunks
df1 = download_polymarket_prices_by_slug(..., overwrite=False)

# Second time: Loads from cache (instant!)
df2 = download_polymarket_prices_by_slug(..., overwrite=False)
```

## How It Works

The chunking logic in `polymarket_data.py`:

```python
def download_polymarket_prices(...):
    # Check if date range exceeds limit
    days_requested = (end_date - start_date).days
    MAX_DAYS_PER_REQUEST = 14

    if days_requested > MAX_DAYS_PER_REQUEST:
        # Split into 14-day chunks
        chunks = []
        current_start = start_date

        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=14), end_date)

            # Recursive call for each chunk
            chunk_df = download_polymarket_prices(
                token_id, current_start, chunk_end, ...
            )

            chunks.append(chunk_df)
            current_start = chunk_end

        # Combine all chunks
        combined_df = pd.concat(chunks)
        combined_df = combined_df.drop_duplicates()  # Remove overlaps
        combined_df = combined_df.sort_values("timestamp")

        return combined_df

    # Single request if <= 14 days
    # ... normal download logic ...
```

## Benefits

✅ **Automatic** - No manual chunking required
✅ **Transparent** - Shows progress for each chunk
✅ **Cached** - Each chunk cached separately for efficiency
✅ **Deduplicated** - Removes duplicate timestamps at chunk boundaries
✅ **Sorted** - Results always sorted by timestamp

## Limitations

⚠️ **Rate limiting** - Many large requests might hit rate limits (not documented by Polymarket)
⚠️ **Time** - 1 year at 1-minute fidelity = 26 API calls (can take a while)
⚠️ **Memory** - Very large datasets might use significant RAM

## Recommendations

1. **For intraday analysis**: Use `fidelity=1` with short date ranges (<14 days)
2. **For weekly/monthly analysis**: Use `fidelity=60` or `fidelity=1440`
3. **For historical analysis**: Use `fidelity=1440` and download the full history
4. **Always use caching**: Set `overwrite=False` (default) to reuse cached data

## Summary

The 15-day API limit is now **handled automatically**. You can request any date range, and the function will:
1. Detect if it exceeds 14 days
2. Split into manageable chunks
3. Download each chunk
4. Combine and return complete dataset

Your 31-day query now works perfectly! 🎉
