# Understanding the `fidelity` Parameter

## What is `fidelity`?

The `fidelity` parameter controls the **time interval** between data points returned by the Polymarket CLOB API.

- `fidelity=1` → One data point per **minute**
- `fidelity=5` → One data point every **5 minutes**
- `fidelity=60` → One data point per **hour**
- `fidelity=1440` → One data point per **day**

## Your Issue: Why Only 1 Row?

You queried December 1-2, 2024, but the market was **created on December 29, 2024**!

```python
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    start_date=datetime(2024, 12, 1),  # ← Market didn't exist yet!
    end_date=datetime(2024, 12, 2),
    fidelity=1
)
# Result: 1 row (the most recent price at query time)
```

The API returns the current price when no historical data exists for the requested range.

## Solution: Query After Market Creation

```python
from polymarket_data import download_polymarket_prices_by_slug
from datetime import datetime

# Market was created Dec 29, 2024
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    start_date=datetime(2024, 12, 30),  # ✓ After creation
    end_date=datetime(2024, 12, 31),
    fidelity=1,  # Minute-by-minute
    overwrite=True  # Force fresh download (ignore stale cache)
)

print(f"Downloaded {len(df)} data points")
# Output: Downloaded 1441 data points (one per minute)
```

## How to Find When a Market Was Created

```python
from polymarket_data import get_market_info

market = get_market_info("fed-rate-hike-in-2025")
print(f"Created: {market['createdAt']}")
print(f"Start Date: {market['startDate']}")
print(f"End Date: {market['endDate']}")
```

Output:
```
Created: 2024-12-29T17:38:00.916304Z
Start Date: 2024-12-29T22:50:33.584839Z
End Date: 2025-12-10T12:00:00Z
```

## Expected Data Points for Different Fidelities

For a **1-day period** (24 hours):

| Fidelity | Interval | Expected Points | Example |
|----------|----------|-----------------|---------|
| 1        | 1 minute | ~1,440          | Minute bars |
| 5        | 5 minutes | ~288           | 5-min bars |
| 60       | 1 hour   | ~24            | Hourly bars |
| 1440     | 1 day    | ~1             | Daily bars |

For a **1-week period** (7 days):

| Fidelity | Interval | Expected Points |
|----------|----------|-----------------|
| 1        | 1 minute | ~10,080        |
| 5        | 5 minutes | ~2,016        |
| 60       | 1 hour   | ~168          |
| 1440     | 1 day    | ~7            |

## Important Notes

### 1. Data Only Exists After Market Creation

Markets have a `startDate`. No data exists before this date.

### 2. Cache Can Be Stale

If you previously downloaded data for a date range with no data, you'll get a cached empty/minimal result:

```python
# Force fresh download
df = download_polymarket_prices_by_slug(
    ...,
    overwrite=True  # ← Ignore cache, download fresh
)
```

### 3. API Returns Most Recent Price for Invalid Ranges

When you query a date range with no data, the API returns the current market price:

```python
# Query future date
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    start_date=datetime(2026, 1, 1),  # Future!
    end_date=datetime(2026, 1, 2),
    fidelity=1
)
# Result: 1 row with current price
```

### 4. Not All Minutes Have Trades

Even with `fidelity=1`, you might not get exactly 1,440 points per day because:
- The API returns the last known price at each interval
- If there were no trades in a minute, the API might skip it
- Market hours might not be 24/7

## Real Example: Minute-by-Minute Data

```python
from polymarket_data import download_polymarket_prices_by_slug
from datetime import datetime

# Download minute data for one day
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    start_date=datetime(2024, 12, 30),
    end_date=datetime(2024, 12, 31),
    fidelity=1,  # 1-minute intervals
    overwrite=True
)

print(f"Downloaded {len(df)} data points")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nTime between points:")
print(df["timestamp"].diff().describe())
```

Output:
```
Downloaded 1441 data points

First 5 rows:
            timestamp  price  unix_timestamp
0 2024-12-30 06:00:04  0.155      1735538404
1 2024-12-30 06:01:03  0.155      1735538463
2 2024-12-30 06:02:03  0.155      1735538523
3 2024-12-30 06:03:03  0.155      1735538583
4 2024-12-30 06:04:03  0.155      1735538643

Time between points:
count                        1440
mean     0 days 00:00:59.993...
std      0 days 00:00:00.078...
min      0 days 00:00:48
max      0 days 00:01:11
```

## Recommendations

### For High-Frequency Analysis
```python
fidelity=1  # Minute bars (most detailed)
```

### For Hourly Analysis
```python
fidelity=60  # Hourly bars (recommended for most use cases)
```

### For Daily Analysis
```python
fidelity=1440  # Daily bars
```

### For Long Time Ranges
```python
# Downloading 1 year of minute data = 525,600 points
# Instead, use hourly or daily:
fidelity=60  # or 1440
```

## Checking Data Availability

Before downloading, check when the market was created:

```python
from polymarket_data import get_market_info
from datetime import datetime

market = get_market_info("your-market-slug")
start_date = datetime.fromisoformat(market["startDate"].replace("Z", "+00:00"))

print(f"Market started: {start_date}")
print(f"Data available from: {start_date.date()}")

# Now query AFTER this date
df = download_polymarket_prices_by_slug(
    market_slug="your-market-slug",
    start_date=start_date,  # ✓ Use actual start date
    end_date=datetime.now(),
    fidelity=60
)
```

## Troubleshooting

### "I get way fewer points than expected"

**Possible causes:**
1. Market didn't exist for entire date range
2. Using cached data with `overwrite=False`
3. Requesting future dates

**Solution:** Use `overwrite=True` and query dates after market creation

### "I get only 1 data point"

**Cause:** Market didn't exist during requested date range

**Solution:** Check market creation date and query after that:
```python
market = get_market_info("market-slug")
print(f"Created: {market['createdAt']}")
```
