# Polymarket Data Downloader - Slug to Token ID Mapping

## Overview

This document explains how to map Polymarket slugs to token IDs and download historical price data.

## Key Concepts

### Slugs
- **Market Slug**: URL-friendly identifier for individual markets (e.g., `"fed-rate-hike-in-2025"`)
- **Event Slug**: URL-friendly identifier for events containing multiple related markets (e.g., `"presidential-election-winner-2024"`)

### Token IDs
- CLOB token IDs are large integers (as strings) that uniquely identify specific market outcomes
- Each binary market typically has 2 token IDs: one for "Yes" and one for "No"
- Token IDs are required to download historical price data from the CLOB API

### Events vs Markets
- **Event**: A collection of related markets (e.g., "Presidential Election Winner 2024")
  - Contains multiple markets, one for each candidate
  - URL format: `https://polymarket.com/event/{event-slug}`

- **Market**: A single prediction market with outcomes (e.g., "Will the Fed raise rates?")
  - URL format: `https://polymarket.com/event/{market-slug}` (note: same format!)

**Important**: The URL `https://polymarket.com/event/presidential-election-winner-2024` refers to an **EVENT**, not a single market!

## Functions

### `slug_to_token_ids(slug, is_event=False)`

Maps a slug to token IDs for all associated markets.

**Parameters:**
- `slug` (str): The market or event slug
- `is_event` (bool): Set to `True` for events, `False` for individual markets

**Returns:**
- Dictionary mapping candidate/market names to lists of token IDs

**Example - Event (Presidential Election):**
```python
from polymarket_data import slug_to_token_ids

# Get all token IDs for presidential election
result = slug_to_token_ids("presidential-election-winner-2024", is_event=True)

# Result structure:
# {
#   "Donald Trump": ["21742633143463906290569050155826241533067272736897614950488156847949938836455", "..."],
#   "Kamala Harris": ["69236923620077691027083946871148646972011131466059644796654161903044970987404", "..."],
#   ...
# }

# Get Trump's "Yes" token ID
trump_yes_token = result["Donald Trump"][0]
trump_no_token = result["Donald Trump"][1]
```

**Example - Single Market:**
```python
# Get token IDs for a single market
result = slug_to_token_ids("fed-rate-hike-in-2025", is_event=False)

# Result structure:
# {
#   "Will the Fed raise rates in 2025?": ["token_id_yes", "token_id_no"]
# }
```

### `get_event_info(event_slug)`

Get detailed information about an event and all its markets.

**Example:**
```python
from polymarket_data import get_event_info

event = get_event_info("presidential-election-winner-2024")
print(event['title'])  # "Presidential Election Winner 2024"
print(len(event['markets']))  # Number of candidate markets
```

### `get_market_info(market_slug, max_pages=10)`

Get detailed information about a single market.

**Example:**
```python
from polymarket_data import get_market_info

market = get_market_info("fed-rate-hike-in-2025")
print(market['question'])
print(market['clobTokenIds'])  # Token IDs as JSON string
```

## Complete Example: Presidential Election 2024

### Method 1: Using `download_polymarket_prices_by_event()` (EASIEST)

```python
from datetime import datetime
from polymarket_data import download_polymarket_prices_by_event

# Download Trump's data directly by candidate name
df = download_polymarket_prices_by_event(
    event_slug="presidential-election-winner-2024",
    candidate_name="Donald Trump",
    outcome_index=0,  # "Yes" token
    start_date=datetime(2024, 10, 28),
    end_date=datetime(2024, 11, 5),
    fidelity=60  # Hourly data
)

print(df.head())
```

### Method 2: Using `slug_to_token_ids()` then downloading

```python
from datetime import datetime
from polymarket_data import slug_to_token_ids, download_polymarket_prices

# 1. Get all token IDs
token_map = slug_to_token_ids("presidential-election-winner-2024", is_event=True)

# 2. Print all candidates
for candidate, tokens in token_map.items():
    print(f"{candidate}: {tokens[0]}")

# 3. Download historical data for Trump
trump_token = token_map["Donald Trump"][0]  # "Yes" token

df = download_polymarket_prices(
    token_id=trump_token,
    start_date=datetime(2024, 10, 28),  # Week before election
    end_date=datetime(2024, 11, 5),     # Election day
    fidelity=60  # Hourly data
)

print(df.head())
```

## Presidential Election 2024 - Token IDs

For convenience, here are the token IDs for all candidates in the 2024 presidential election event:

| Candidate | Yes Token ID |
|-----------|-------------|
| Donald Trump | `21742633143463906290569050155826241533067272736897614950488156847949938836455` |
| Kamala Harris | `69236923620077691027083946871148646972011131466059644796654161903044970987404` |
| Joe Biden | `88027839609243624193415614179328679602612916497045596227438675518749602824929` |
| Robert F. Kennedy Jr. | `75551890681049796405776295654438099776333571510662809052054780589218524237663` |
| Ron DeSantis | `54541905023211985194827443687227462634594584372996482268933020846517872533280` |
| Nikki Haley | `19083349462791593334532840548890602187185739923311385087650426802477691161360` |
| Vivek Ramaswamy | `71118168890902402346450607953977430866499056452499149647300109878547888435163` |
| Gavin Newsom | `99200347365169760700385453164878188504479548439905371494493482364634358863823` |
| Michelle Obama | `97508453625137094121006941885029334584603955750917059456402541591996493525667` |
| Hillary Clinton | `79316691944049488812500733050438507204613781002222375264046442941003895009475` |
| Bernie Sanders | `95128817762909535143571435260705470642391662537976312011260538371392879420759` |
| Elizabeth Warren | `6025348680810459235592257487856478394037580571221769223427710907585587056389` |
| Chris Christie | `27312896015258311102305871640185491718068302146240154758497460598552961305988` |
| AOC | `6238317280296426865475638559260472448644617115418089359113344407432348159324` |
| Kanye | `48285207411891694847413807268670593735244327770017422161322089036370055854362` |
| Other Democrat Politician | `74706296939809671893768905246606398708802232875822379413753245164957842209130` |
| Other Republican Politician | `87935798830831555521299232238121934560977823768906296045917813721531790174443` |

## API Endpoints

The module uses two Polymarket APIs:

1. **Gamma Markets API** (`https://gamma-api.polymarket.com/markets`)
   - Lists all individual markets
   - Searchable by slug (requires pagination)

2. **Gamma Events API** (`https://gamma-api.polymarket.com/events`)
   - Lists events (collections of markets)
   - Directly queryable by slug parameter

3. **CLOB API** (`https://clob.polymarket.com/prices-history`)
   - Downloads historical price data by token ID

## Important Notes & Limitations

### When to Use Each Function

1. **For EVENTS (like presidential election)**: Use `download_polymarket_prices_by_event()`
   - Fast and easy
   - Events API allows direct lookup by slug
   - Example: Presidential election with multiple candidates

2. **For INDIVIDUAL MARKETS**: Use `download_polymarket_prices_by_slug()` ✨ NOW FAST!
   - **NEW**: Now uses direct API slug query (instant, not slow pagination)
   - Works for ALL markets, not just first 10K
   - Recommended for simple markets
   - Example: "fed-rate-hike-in-2025", "us-recession-in-2025"

3. **For MAXIMUM SPEED**: Use token IDs directly
   - If you already have the token ID, use `download_polymarket_prices()` directly
   - Saves one API call to look up the slug
   - Use when you've cached token IDs or know them in advance

### Key Facts

- The `TEST_MARKET_SLUG` in the test file is `"fed-rate-hike-in-2025"` - this is a **single market** slug
- The presidential election slug `"presidential-election-winner-2024"` is an **event** slug (not a market)
- Yes, you can map from slug to token_id! Use `slug_to_token_ids()` with the appropriate `is_event` flag
- Event slugs can be queried directly via the Events API (fast)
- **Market slugs now support direct queries too!** ✨ (fast as of latest update)
- Old pagination fallback still exists for robustness
