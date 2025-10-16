"""
Unit tests for fetching closed Polymarket markets metadata.

Tests the polymarket_data module's ability to:
- Fetch closed markets from Gamma API
- Cache results to disk
- Reload from cache efficiently
"""

import pytest
import pandas as pd
from pathlib import Path
import shutil

from polymarket_data import get_all_closed_markets


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_polymarket_data"
    data_dir.mkdir(exist_ok=True)
    yield str(data_dir)
    # Cleanup
    if data_dir.exists():
        shutil.rmtree(data_dir)


class TestClosedMarketsMetadata:
    """Test suite for closed markets metadata functionality."""

    def test_fetch_limited_markets(self, test_data_dir):
        """Test fetching a limited number of closed markets."""
        # Fetch just 50 markets to keep test fast
        df = get_all_closed_markets(cache_dir=test_data_dir, max_markets=50)

        # Validate DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 50

        # Check for essential columns
        assert "slug" in df.columns
        assert "question" in df.columns
        assert "volume_num" in df.columns
        assert "liquidity_num" in df.columns
        assert "end_date_parsed" in df.columns
        assert "closed_time_parsed" in df.columns
        assert "outcomes" in df.columns
        assert "category" in df.columns

        # Validate data types
        assert pd.api.types.is_float_dtype(df["volume_num"])
        assert pd.api.types.is_float_dtype(df["liquidity_num"])
        assert pd.api.types.is_datetime64_any_dtype(df["end_date_parsed"])

        # Validate markets are actually closed
        assert df["closed"].all()

        print(f"\n✓ Fetched {len(df)} closed markets")
        print(f"  Columns: {len(df.columns)}")
        print(
            f"  Volume range: ${df['volume_num'].min():.2f} - ${df['volume_num'].max():.2f}"
        )

    def test_caching_mechanism(self, test_data_dir):
        """Test that caching works correctly."""
        # First fetch - should hit API
        df1 = get_all_closed_markets(cache_dir=test_data_dir, max_markets=25)

        # Check cache file exists
        cache_file = Path(test_data_dir) / "closed_markets_metadata.parquet"
        assert cache_file.exists()

        # Get file modification time
        mtime1 = cache_file.stat().st_mtime

        # Second fetch - should load from cache
        df2 = get_all_closed_markets(cache_dir=test_data_dir, max_markets=25)

        # File should not be modified
        mtime2 = cache_file.stat().st_mtime
        assert mtime1 == mtime2

        # DataFrames should be identical in content
        # Note: Parquet may change some data types (e.g., bool vs object)
        # so we check important columns rather than full frame equality
        assert len(df1) == len(df2)
        assert list(df1.columns) == list(df2.columns)
        assert (df1["slug"] == df2["slug"]).all()
        assert (df1["question"] == df2["question"]).all()
        assert (df1["volume_num"] == df2["volume_num"]).all()

        print("\n✓ Caching mechanism working correctly")

    def test_overwrite_flag(self, test_data_dir):
        """Test that overwrite=True re-fetches data."""
        # First fetch
        df1 = get_all_closed_markets(
            cache_dir=test_data_dir, max_markets=25, overwrite=False
        )

        cache_file = Path(test_data_dir) / "closed_markets_metadata.parquet"
        mtime1 = cache_file.stat().st_mtime

        # Small delay to ensure different modification time
        import time

        time.sleep(0.1)

        # Fetch with overwrite=True
        df2 = get_all_closed_markets(
            cache_dir=test_data_dir, max_markets=25, overwrite=True
        )

        # File should be modified
        mtime2 = cache_file.stat().st_mtime
        assert mtime2 > mtime1

        print("\n✓ Overwrite flag working correctly")

    def test_data_quality(self, test_data_dir):
        """Test that fetched data has expected quality."""
        df = get_all_closed_markets(cache_dir=test_data_dir, max_markets=50)

        # All markets should be closed
        assert df["closed"].all()

        # Volume should be non-negative
        assert (df["volume_num"] >= 0).all()

        # Liquidity should be non-negative
        assert (df["liquidity_num"] >= 0).all()

        # End dates should be in the past (excluding NaT values)
        valid_end_dates = df["end_date_parsed"].dropna()
        if len(valid_end_dates) > 0:
            assert (valid_end_dates < pd.Timestamp.now(tz="UTC")).all()

        # Questions should not be empty
        assert df["question"].notna().all()
        assert (df["question"].str.len() > 0).all()

        # Slugs should not be empty
        assert df["slug"].notna().all()
        assert (df["slug"].str.len() > 0).all()

        print("\n✓ Data quality checks passed")

    def test_filtering_high_volume_markets(self, test_data_dir):
        """Test that we can filter for high-volume markets."""
        df = get_all_closed_markets(cache_dir=test_data_dir, max_markets=100)

        # Filter for markets with volume > $100k
        high_volume = df[df["volume_num"] > 100_000]

        assert len(high_volume) > 0
        assert (high_volume["volume_num"] > 100_000).all()

        print(f"\n✓ Found {len(high_volume)} markets with >$100k volume")
        print(f"  Top 5 by volume:")
        top5 = high_volume.nlargest(5, "volume_num")[["question", "volume_num"]]
        for idx, row in top5.iterrows():
            print(f"    ${row['volume_num']:,.0f} - {row['question'][:60]}...")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
