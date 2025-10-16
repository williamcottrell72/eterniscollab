# Quick Start: Polymarket Data Download

## TL;DR - Just Show Me The Code!

### Step 1: Explore Available Markets

```python
from polymarket_data import get_event_markets

# Get all markets in an event
markets = get_event_markets("presidential-election-winner-2024")

# Print all available markets
for market_id, info in markets.items():
    print(f"{market_id}: {info['question']}")
```

### Step 2: Download Data for a Specific Market

```python
from polymarket_data import download_polymarket_prices_by_event
from datetime import datetime

# Download Trump's odds
df = download_polymarket_prices_by_event(
    event_slug="presidential-election-winner-2024",
    market_id="Donald Trump",  # Use market_id from get_event_markets()
    start_date=datetime(2024, 10, 28),
    end_date=datetime(2024, 11, 5),
    fidelity=60  # Hourly data (use 1 for minute data)
)

print(df.head())
```

### One-Liner: List All Markets

```python
from polymarket_data import get_event_markets

markets = get_event_markets("presidential-election-winner-2024")
print(list(markets.keys()))
```

**Outputs:**
- Donald Trump
- Kamala Harris
- Joe Biden
- Robert F. Kennedy Jr.
- Ron DeSantis
- Nikki Haley
- Vivek Ramaswamy
- Gavin Newsom
- Michelle Obama
- Hillary Clinton
- Bernie Sanders
- Elizabeth Warren
- Chris Christie
- AOC
- Kanye
- Other Democrat Politician
- Other Republican Politician

### Download Using Token ID Directly (Fastest)

If you already know the token ID:

```python
from polymarket_data import download_polymarket_prices
from datetime import datetime

# Trump's "Yes" token ID
token_id = "21742633143463906290569050155826241533067272736897614950488156847949938836455"

df = download_polymarket_prices(
    token_id=token_id,
    start_date=datetime(2024, 10, 28),
    end_date=datetime(2024, 11, 5),
    fidelity=60  # Hourly
)
```

## Common Token IDs

| Candidate | Yes Token ID |
|-----------|-------------|
| Donald Trump | `21742633143463906290569050155826241533067272736897614950488156847949938836455` |
| Kamala Harris | `69236923620077691027083946871148646972011131466059644796654161903044970987404` |

See `presidential_election_2024_tokens.json` for the complete list.

## Parameters Explained

- **`fidelity`**: Time resolution in minutes
  - `1` = minute-level data (most detailed)
  - `60` = hourly data (recommended for large date ranges)
  - `1440` = daily data

- **`outcome_index`**:
  - `0` = "Yes" token (probability market resolves to "Yes")
  - `1` = "No" token (probability market resolves to "No")

- **`overwrite`**:
  - `False` (default) = Use cached data if available
  - `True` = Re-download even if cached

## Error: "Market slug not found"?

If you get `ValueError: Market with slug 'xxx' not found`:

1. Check if it's an **EVENT** (use `download_polymarket_prices_by_event()`)
2. If it's a market, the slug might not be in the first 10,000 markets
3. Find the token ID manually from the Polymarket website and use `download_polymarket_prices()` directly

## Full Documentation

See `POLYMARKET_README.md` for complete documentation and all available functions.
