"""
Unit tests for utility functions.

Tests the shared utility functions used across multiple modules.
"""

import pytest
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import extract_numeric_value


class TestExtractNumericValue:
    """Test the unified numeric value extraction function."""

    def test_extract_percentage(self):
        """Test extracting values from percentage format."""
        assert extract_numeric_value("The value is 25%") == 0.25
        assert extract_numeric_value("I estimate 75.5%") == 0.755
        assert extract_numeric_value("100%") == 1.0
        assert extract_numeric_value("0%") == 0.0

    def test_extract_decimal(self):
        """Test extracting values from decimal format."""
        assert extract_numeric_value("Value: 0.25") == 0.25
        assert extract_numeric_value("The answer is 0.75") == 0.75
        assert extract_numeric_value("0.5") == 0.5
        assert extract_numeric_value("0.333") == 0.333

    def test_extract_fraction(self):
        """Test extracting values from fraction format."""
        assert extract_numeric_value("1/4 chance") == 0.25
        assert extract_numeric_value("3/4 probability") == 0.75
        assert extract_numeric_value("1/2") == 0.5
        assert extract_numeric_value("2 / 5") == 0.4

    def test_extract_edge_cases(self):
        """Test edge cases like 0 and 1."""
        assert extract_numeric_value("0") == 0.0
        assert extract_numeric_value("1") == 1.0
        assert extract_numeric_value("0.0") == 0.0
        assert extract_numeric_value("1.0") == 1.0

    def test_custom_value_name(self):
        """Test that custom value names appear in error messages."""
        with pytest.raises(ValueError) as exc_info:
            extract_numeric_value("No number here", "custom_metric")
        assert "custom_metric" in str(exc_info.value)

    def test_multiple_values_returns_first(self):
        """Test that when multiple values exist, the first valid one is returned."""
        # Fraction has priority
        assert extract_numeric_value("1/2 or 75%") == 0.5

        # If no fractions, percentage comes next
        assert extract_numeric_value("25% or 0.8") == 0.25

        # If no percentages or fractions, decimal is used
        assert extract_numeric_value("0.3 or 0.7") == 0.3

    def test_invalid_text(self):
        """Test that invalid text raises ValueError."""
        with pytest.raises(ValueError):
            extract_numeric_value("No value here")

        with pytest.raises(ValueError):
            extract_numeric_value("The weather is nice")

    def test_out_of_range_ignored(self):
        """Test that values outside 0-1 range are ignored."""
        # Only the valid 0.5 should be extracted
        assert extract_numeric_value("150% is too high, use 0.5") == 0.5

        # Should raise error if no valid values
        with pytest.raises(ValueError):
            extract_numeric_value("150% probability")

    def test_various_formats(self):
        """Test various real-world response formats."""
        # Labeled formats
        assert extract_numeric_value("Probability: 0.45") == 0.45
        assert extract_numeric_value("Score: 60%") == 0.6
        assert extract_numeric_value("Estimated at 3/4") == 0.75

        # Sentence formats
        assert extract_numeric_value("I estimate the probability to be around 30%") == 0.3
        assert extract_numeric_value("The score would be approximately 0.65") == 0.65
        assert extract_numeric_value("About 1/3 of the time") == pytest.approx(0.333, abs=0.001)

        # Multiple formats in one response
        assert extract_numeric_value("While some say 80%, I think 0.6 is more accurate") == 0.8


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
