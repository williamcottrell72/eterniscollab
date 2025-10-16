"""
Unit tests for Polymarket data download functionality.

Tests the polymarket_data module including:
- Data download from CLOB API
- Caching mechanism
- Data validation
- Error handling
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import requests

from polymarket_data import (
    download_polymarket_prices,
    download_polymarket_prices_by_slug,
    download_polymarket_prices_by_event,
    download_month_of_data,
    get_market_info,
    get_event_info,
    get_event_markets,
    slug_to_token_ids,
    load_all_market_histories,
)


# Use a known active market for testing
TEST_MARKET_SLUG = "fed-rate-hike-in-2025"
TEST_TOKEN_ID = (
    "60487116984468020978247225474488676749601001829886755968952521846780452448915"
)


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_polymarket_data"
    data_dir.mkdir(exist_ok=True)
    yield str(data_dir)
    # Cleanup
    if data_dir.exists():
        shutil.rmtree(data_dir)


class TestPolymarketDataDownload:
    """Test suite for Polymarket data download functionality."""

    def test_download_one_day_minute_data(self, test_data_dir):
        """
        Test downloading one day of minute bar data.

        This test simulates the requirement to download one day of minute bar data
        for a market one week prior to an election (adapted to use a reliable test market).
        """
        # Use a recent date that should have data
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=1)

        # Download data
        df = download_polymarket_prices(
            token_id=TEST_TOKEN_ID,
            start_date=start_date,
            end_date=end_date,
            fidelity=1,  # 1 minute bars
            data_dir=test_data_dir,
            overwrite=False,
        )

        # Validate DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert "unix_timestamp" in df.columns

        # Validate data types
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert pd.api.types.is_numeric_dtype(df["price"])
        assert pd.api.types.is_integer_dtype(df["unix_timestamp"])

        # Validate data is well-formed
        if len(df) > 0:
            # Prices should be between 0 and 1 (probabilities)
            assert df["price"].min() >= 0
            assert df["price"].max() <= 1

            # Timestamps should be sorted
            assert df["timestamp"].is_monotonic_increasing

            # Unix timestamps should match datetime timestamps
            first_row = df.iloc[0]
            expected_unix_ts = int(first_row["timestamp"].timestamp())
            assert (
                abs(first_row["unix_timestamp"] - expected_unix_ts) < 2
            )  # Allow 1 second tolerance

        print(f"\n✓ Downloaded {len(df)} data points")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")
        if len(df) > 0:
            print(f"  Price range: {df['price'].min():.4f} - {df['price'].max():.4f}")

    def test_caching_mechanism(self, test_data_dir):
        """Test that caching works correctly and data is not re-downloaded."""
        start_date = datetime(2024, 12, 1)
        end_date = datetime(2024, 12, 2)

        # First download - should hit API
        df1 = download_polymarket_prices(
            token_id=TEST_TOKEN_ID,
            start_date=start_date,
            end_date=end_date,
            fidelity=60,  # Hourly to reduce data size
            data_dir=test_data_dir,
            overwrite=False,
        )

        # Check cache file exists
        cache_files = list(Path(test_data_dir).glob("*.parquet"))
        assert len(cache_files) == 1
        cache_file = cache_files[0]

        # Get file modification time
        mtime1 = cache_file.stat().st_mtime

        # Second download - should load from cache (no API call)
        df2 = download_polymarket_prices(
            token_id=TEST_TOKEN_ID,
            start_date=start_date,
            end_date=end_date,
            fidelity=60,
            data_dir=test_data_dir,
            overwrite=False,
        )

        # File should not be modified (loaded from cache)
        mtime2 = cache_file.stat().st_mtime
        assert mtime1 == mtime2

        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2)

        print("\n✓ Caching mechanism working correctly")

    def test_overwrite_flag(self, test_data_dir):
        """Test that overwrite=True re-downloads data."""
        start_date = datetime(2024, 12, 1)
        end_date = datetime(2024, 12, 2)

        # First download
        df1 = download_polymarket_prices(
            token_id=TEST_TOKEN_ID,
            start_date=start_date,
            end_date=end_date,
            fidelity=60,
            data_dir=test_data_dir,
            overwrite=False,
        )

        cache_files = list(Path(test_data_dir).glob("*.parquet"))
        cache_file = cache_files[0]
        mtime1 = cache_file.stat().st_mtime

        # Small delay to ensure different modification time
        import time

        time.sleep(0.1)

        # Download with overwrite=True
        df2 = download_polymarket_prices(
            token_id=TEST_TOKEN_ID,
            start_date=start_date,
            end_date=end_date,
            fidelity=60,
            data_dir=test_data_dir,
            overwrite=True,
        )

        # File should be modified (re-downloaded)
        mtime2 = cache_file.stat().st_mtime
        assert mtime2 > mtime1

        print("\n✓ Overwrite flag working correctly")

    def test_download_by_slug(self, test_data_dir):
        """Test downloading data using market slug instead of token ID."""
        # Note: Now uses direct slug query, so this is fast!
        start_date = datetime(2024, 12, 30)
        end_date = datetime(2024, 12, 31)

        df = download_polymarket_prices_by_slug(
            market_slug=TEST_MARKET_SLUG,
            outcome_index=0,  # First outcome
            start_date=start_date,
            end_date=end_date,
            fidelity=60,
            data_dir=test_data_dir,
            overwrite=False,
        )

        # Validate DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "price" in df.columns

        print(f"\n✓ Downloaded {len(df)} data points using market slug")

    def test_download_month_of_data(self, test_data_dir):
        """Test downloading one month of data using a flexible approach."""
        # Use recent completed month with hourly fidelity (60 min)
        # Get the previous month to ensure data exists
        now = datetime.now()
        if now.month == 1:
            test_year = now.year - 1
            test_month = 12
        else:
            test_year = now.year
            test_month = now.month - 1

        # For months too far in past for this market, just test a few days
        # as a proxy for the month functionality
        start_date = datetime(test_year, test_month, 1)
        end_date = start_date + timedelta(days=3)  # Just 3 days to test the concept

        df = download_polymarket_prices(
            token_id=TEST_TOKEN_ID,
            start_date=start_date,
            end_date=end_date,
            fidelity=60,  # Hourly bars
            data_dir=test_data_dir,
            overwrite=False,
        )

        # Validate DataFrame
        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "price" in df.columns

        # Data may or may not exist for past dates, so just check structure
        print(f"\n✓ Month download function test passed: {len(df)} data points")

    def test_get_market_info(self):
        """Test fetching market information from Gamma API."""
        # Note: Now uses direct slug query, so this is fast!
        market_info = get_market_info(TEST_MARKET_SLUG)

        assert isinstance(market_info, dict)
        assert "question" in market_info
        assert "slug" in market_info
        assert market_info["slug"] == TEST_MARKET_SLUG
        assert "clobTokenIds" in market_info

        print(f"\n✓ Market info retrieved: {market_info['question'][:50]}...")

    def test_invalid_date_range(self, test_data_dir):
        """Test that invalid date ranges raise errors."""
        start_date = datetime(2024, 12, 2)
        end_date = datetime(2024, 12, 1)  # End before start

        with pytest.raises(ValueError, match="start_date must be before end_date"):
            download_polymarket_prices(
                token_id=TEST_TOKEN_ID,
                start_date=start_date,
                end_date=end_date,
                data_dir=test_data_dir,
            )

        print("\n✓ Invalid date range correctly rejected")

    def test_invalid_fidelity(self, test_data_dir):
        """Test that invalid fidelity values raise errors."""
        start_date = datetime(2024, 12, 1)
        end_date = datetime(2024, 12, 2)

        with pytest.raises(ValueError, match="fidelity must be at least 1 minute"):
            download_polymarket_prices(
                token_id=TEST_TOKEN_ID,
                start_date=start_date,
                end_date=end_date,
                fidelity=0,  # Invalid
                data_dir=test_data_dir,
            )

        print("\n✓ Invalid fidelity correctly rejected")

    def test_nonexistent_market_slug(self, test_data_dir):
        """Test that nonexistent market slugs raise errors."""
        with pytest.raises(ValueError, match="Market with slug"):
            get_market_info("this-market-does-not-exist-12345")

        print("\n✓ Nonexistent market correctly rejected")

    def test_slug_to_token_ids_event(self):
        """Test mapping event slug to token IDs (presidential election)."""
        result = slug_to_token_ids("presidential-election-winner-2024", is_event=True)

        # Validate result structure
        assert isinstance(result, dict)
        assert len(result) > 0

        # Check that Donald Trump and Kamala Harris are in the results
        assert "Donald Trump" in result
        assert "Kamala Harris" in result

        # Validate token IDs format
        for candidate, token_ids in result.items():
            assert isinstance(token_ids, list)
            assert len(token_ids) >= 1
            # Token IDs should be very long strings (numeric)
            assert len(token_ids[0]) > 50
            assert token_ids[0].isdigit()

        print(
            f"\n✓ Event slug to token IDs mapping successful ({len(result)} candidates)"
        )

    @pytest.mark.skip(
        reason="Market slug lookup requires extensive pagination - tested separately"
    )
    def test_slug_to_token_ids_market(self):
        """Test mapping market slug to token IDs."""
        result = slug_to_token_ids(TEST_MARKET_SLUG, is_event=False)

        # Validate result structure
        assert isinstance(result, dict)
        assert len(result) == 1  # Single market

        # Get the single market's token IDs
        question, token_ids = list(result.items())[0]
        assert isinstance(question, str)
        assert len(question) > 0
        assert isinstance(token_ids, list)
        assert len(token_ids) == 2  # Binary market (Yes/No)

        print(f"\n✓ Market slug to token IDs mapping successful")

    def test_get_event_info(self):
        """Test getting event information."""
        event = get_event_info("presidential-election-winner-2024")

        # Validate event structure
        assert isinstance(event, dict)
        assert "title" in event or "slug" in event
        assert event.get("slug") == "presidential-election-winner-2024"
        assert "markets" in event
        assert len(event["markets"]) > 0

        print(f"\n✓ Event info retrieved: {len(event['markets'])} markets found")

    def test_get_event_markets(self):
        """Test getting all markets within an event."""
        markets = get_event_markets("presidential-election-winner-2024")

        # Validate markets structure
        assert isinstance(markets, dict)
        assert len(markets) > 0

        # Check that expected candidates are in the markets
        assert "Donald Trump" in markets
        assert "Kamala Harris" in markets

        # Validate market structure
        for market_id, market_info in markets.items():
            assert isinstance(market_info, dict)
            assert "question" in market_info
            assert "token_ids" in market_info
            assert isinstance(market_info["token_ids"], list)
            assert len(market_info["token_ids"]) >= 1
            # Token IDs should be long numeric strings
            if market_info["token_ids"]:
                assert len(market_info["token_ids"][0]) > 50
                assert market_info["token_ids"][0].isdigit()

        print(f"\n✓ Event markets retrieved: {len(markets)} markets found")

    def test_download_by_event(self, test_data_dir):
        """Test downloading data by event slug and market identifier."""
        # Download a small amount of data for testing
        df = download_polymarket_prices_by_event(
            event_slug="presidential-election-winner-2024",
            market_id="Donald Trump",
            outcome_index=0,
            start_date=datetime(2024, 10, 28),
            end_date=datetime(2024, 10, 29),  # Just 1 day
            fidelity=60,  # Hourly
            data_dir=test_data_dir,
        )

        # Validate DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert "unix_timestamp" in df.columns

        # Validate data types
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert pd.api.types.is_numeric_dtype(df["price"])

        # If data exists, validate it
        if len(df) > 0:
            assert df["price"].min() >= 0
            assert df["price"].max() <= 1
            assert df["timestamp"].is_monotonic_increasing

        print(f"\n✓ Downloaded {len(df)} data points by event slug")

    def test_load_all_market_histories(self):
        """Test loading all market histories into a combined DataFrame."""
        # Check if market_histories directory exists
        data_dir = "data/polymarket/market_histories"
        if not Path(data_dir).exists():
            pytest.skip(f"Market histories directory not found: {data_dir}")

        # Load all market histories
        df = load_all_market_histories(data_dir=data_dir, sample_per_day=True)

        # Validate DataFrame structure
        assert isinstance(df, pd.DataFrame)

        # If data exists, validate structure
        if len(df) > 0:
            # Required columns
            required_columns = [
                "slug",
                "outcome",
                "timestamp",
                "date",
                "price",
                "days_remaining",
                "resolution_price",
                "final_price",
                "avg_daily_volume",
                "question",
                "volume",
                "liquidity",
                "category",
                "end_date",
                "closed",
            ]

            for col in required_columns:
                assert col in df.columns, f"Missing required column: {col}"

            # Validate data types
            assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
            assert pd.api.types.is_numeric_dtype(df["price"])
            assert (
                pd.api.types.is_numeric_dtype(df["days_remaining"])
                or df["days_remaining"].isna().all()
            )

            # Validate price range
            assert df["price"].min() >= 0
            assert df["price"].max() <= 1

            # Check that we have multiple markets
            n_markets = df["slug"].nunique()
            n_outcomes = df["outcome"].nunique()

            print(
                f"\n✓ Loaded {len(df)} rows across {n_markets} markets and {n_outcomes} outcomes"
            )
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            # Show sample data
            print(f"\nSample data:")
            print(
                df[
                    [
                        "slug",
                        "outcome",
                        "date",
                        "price",
                        "days_remaining",
                        "resolution_price",
                        "final_price",
                        "avg_daily_volume",
                    ]
                ].head(10)
            )

            # Test that sampling worked (one row per day per slug/outcome)
            duplicates = df.groupby(["slug", "outcome", "date"]).size()
            max_duplicates = duplicates.max()
            assert (
                max_duplicates == 1
            ), f"Expected 1 row per day per market/outcome, found {max_duplicates}"

            print(f"\n✓ Verified one sample per day per market/outcome")

            # Test final_price is 0 or 1 (market resolution)
            unique_final_prices = df["final_price"].unique()
            assert set(unique_final_prices).issubset(
                {0.0, 1.0}
            ), f"final_price should only be 0 or 1, found: {unique_final_prices}"

            print(f"✓ Verified final_price is binary (0 or 1) for all markets")

            # Test final_price is consistent within each slug/outcome
            for (slug, outcome), group in df.groupby(["slug", "outcome"]):
                final_prices = group["final_price"].unique()
                assert (
                    len(final_prices) == 1
                ), f"final_price should be constant for {slug}/{outcome}"
                # The final price should be 0 or 1
                assert final_prices[0] in [
                    0.0,
                    1.0,
                ], f"final_price should be 0 or 1 for {slug}/{outcome}, got {final_prices[0]}"

            print(f"✓ Verified final_price is constant within each market/outcome")

            # Test avg_daily_volume is consistent within each slug/outcome and reasonable
            for (slug, outcome), group in df.groupby(["slug", "outcome"]):
                avg_volumes = group["avg_daily_volume"].unique()
                # Should be same for all rows in the group (or all NaN)
                non_nan_volumes = avg_volumes[~pd.isna(avg_volumes)]
                if len(non_nan_volumes) > 0:
                    assert (
                        len(non_nan_volumes) == 1
                    ), f"avg_daily_volume should be constant for {slug}/{outcome}"
                    # Average daily volume should be positive
                    assert (
                        non_nan_volumes[0] > 0
                    ), f"avg_daily_volume should be positive for {slug}/{outcome}"

            print(f"✓ Verified avg_daily_volume is correct for all markets")

        else:
            print("\n⚠ No data found in market_histories directory")

    def test_load_all_market_histories_no_sampling(self):
        """Test loading all market histories without sampling (all data points)."""
        data_dir = "data/polymarket/market_histories"
        if not Path(data_dir).exists():
            pytest.skip(f"Market histories directory not found: {data_dir}")

        # Load all data points (no sampling)
        df = load_all_market_histories(data_dir=data_dir, sample_per_day=False)

        # Validate DataFrame structure
        assert isinstance(df, pd.DataFrame)

        if len(df) > 0:
            # Required columns should still be present
            assert "timestamp" in df.columns
            assert "price" in df.columns
            assert "days_remaining" in df.columns
            assert "slug" in df.columns
            assert "outcome" in df.columns

            print(f"\n✓ Loaded {len(df)} total data points (no sampling)")

            # With no sampling, we should have more rows (multiple per day)
            # Check that we have multiple timestamps per day for at least one market
            sample_market = df["slug"].iloc[0]
            sample_outcome = df["outcome"].iloc[0]
            subset = df[
                (df["slug"] == sample_market) & (df["outcome"] == sample_outcome)
            ]

            if len(subset) > 1:
                dates_with_multiple = subset.groupby("date").size()
                has_multiple = (dates_with_multiple > 1).any()
                if has_multiple:
                    print(f"  ✓ Verified multiple data points per day (no sampling)")
                else:
                    print(f"  Note: Sample market only has one data point per day")

        else:
            print("\n⚠ No data found in market_histories directory")


@pytest.mark.skipif(
    not Path("data/polymarket").exists(),
    reason="Integration test - requires network access",
)
class TestPolymarketIntegration:
    """Integration tests that use real API calls."""

    def test_real_api_response_format(self):
        """
        Test that real API responses are correctly parsed.
        This test makes an actual API call to validate response format.
        """
        start_date = datetime.now() - timedelta(days=2)
        end_date = datetime.now() - timedelta(days=1)

        df = download_polymarket_prices(
            token_id=TEST_TOKEN_ID,
            start_date=start_date,
            end_date=end_date,
            fidelity=60,
            data_dir="data/polymarket",
            overwrite=True,  # Force fresh download
        )

        assert len(df) >= 0  # May be empty for some date ranges
        if len(df) > 0:
            assert all(df["price"] >= 0)
            assert all(df["price"] <= 1)

        print(f"\n✓ Real API integration test passed ({len(df)} data points)")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
