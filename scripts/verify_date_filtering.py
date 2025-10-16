"""
Verification script to confirm date filtering is working correctly.

This script tests that download_polymarket_prices_by_slug properly filters
data to only include timestamps within the requested date range.
"""

from datetime import datetime
from polymarket_data import download_polymarket_prices_by_slug

# Test parameters
MARKET_SLUG = "fed-rate-hike-in-2025"
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 2, 1)
FIDELITY = 10  # 10-minute intervals

print("=" * 70)
print("VERIFICATION: Date Range Filtering")
print("=" * 70)
print(f"\nMarket: {MARKET_SLUG}")
print(f"Requested range: {START_DATE.date()} to {END_DATE.date()}")
print(f"Fidelity: {FIDELITY} minutes")
print("\nDownloading data (using cache if available)...\n")

# Download data
df = download_polymarket_prices_by_slug(
    market_slug=MARKET_SLUG,
    outcome_index=0,
    start_date=START_DATE,
    end_date=END_DATE,
    fidelity=FIDELITY,
    overwrite=False,  # Use cache
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Verify all timestamps are within range
all_in_range = (df["timestamp"] >= START_DATE).all() and (
    df["timestamp"] < END_DATE
).all()

print(f"\nTotal data points: {len(df)}")
print(f"\nFirst 5 timestamps:")
print(df["timestamp"].head().to_string(index=False))
print(f"\nLast 5 timestamps:")
print(df["timestamp"].tail().to_string(index=False))

print(f"\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(f"\nEarliest timestamp: {df['timestamp'].min()}")
print(f"Latest timestamp:   {df['timestamp'].max()}")
print(f"\nRequested start: {START_DATE}")
print(f"Requested end:   {END_DATE}")

print(f"\n✓ All timestamps >= start_date: {(df['timestamp'] >= START_DATE).all()}")
print(f"✓ All timestamps < end_date:   {(df['timestamp'] < END_DATE).all()}")
print(f"\n{'✅ PASS' if all_in_range else '❌ FAIL'}: All data within requested range")

if not all_in_range:
    # Show any outliers
    outliers = df[(df["timestamp"] < START_DATE) | (df["timestamp"] >= END_DATE)]
    print(f"\n⚠️  Found {len(outliers)} data points outside requested range:")
    print(outliers[["timestamp", "price"]])
else:
    print("\n🎉 Date filtering is working correctly!")
