"""
Unit tests for buzz analyzer functions.

Tests both Claude and OpenAI buzz estimation functions with real API calls.
"""

import pytest
import os
import sys

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buzz import (
    query_openai_interest,
    query_claude_interest,
    query_openai_divisiveness,
    query_claude_divisiveness,
    get_buzz_score,
)
from utils import extract_numeric_value


# Test topics
HIGH_INTEREST_TOPIC = "Artificial Intelligence"
LOW_INTEREST_TOPIC = "Medieval Latin Poetry"
HIGH_DIVISIVE_TOPIC = "2024 US Presidential Election"
LOW_DIVISIVE_TOPIC = "The Beatles"


class TestScoreExtraction:
    """Test the score extraction helper function."""

    def test_extract_percentage(self):
        """Test extracting score from percentage format."""
        assert extract_numeric_value("The score is 25%", "score") == 0.25
        assert extract_numeric_value("I estimate 75.5% level", "score") == 0.755

    def test_extract_decimal(self):
        """Test extracting score from decimal format."""
        assert extract_numeric_value("Score: 0.25", "score") == 0.25
        assert extract_numeric_value("The answer is 0.75", "score") == 0.75
        assert extract_numeric_value("0.5", "score") == 0.5

    def test_extract_fraction(self):
        """Test extracting score from fraction format."""
        assert extract_numeric_value("1/4 of maximum", "score") == 0.25
        assert extract_numeric_value("3/4", "score") == 0.75

    def test_extract_edge_cases(self):
        """Test edge cases like 0 and 1."""
        assert extract_numeric_value("0", "score") == 0.0
        assert extract_numeric_value("1", "score") == 1.0
        assert extract_numeric_value("0.0", "score") == 0.0
        assert extract_numeric_value("1.0", "score") == 1.0

    def test_invalid_text(self):
        """Test that invalid text raises ValueError."""
        with pytest.raises(ValueError):
            extract_numeric_value("No score here at all", "score")


class TestOpenAIInterest:
    """Test OpenAI interest level estimation."""

    @pytest.fixture
    def openai_api_key(self):
        """Get OpenAI API key from environment."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
        return api_key

    def test_query_high_interest_topic(self, openai_api_key):
        """Test OpenAI interest estimation with a high-interest topic."""
        score = query_openai_interest(topic=HIGH_INTEREST_TOPIC, api_key=openai_api_key)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\n{HIGH_INTEREST_TOPIC} - Interest: {score:.3f}")

    def test_query_low_interest_topic(self, openai_api_key):
        """Test OpenAI interest estimation with a low-interest topic."""
        score = query_openai_interest(topic=LOW_INTEREST_TOPIC, api_key=openai_api_key)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\n{LOW_INTEREST_TOPIC} - Interest: {score:.3f}")


class TestClaudeInterest:
    """Test Claude interest level estimation."""

    @pytest.fixture
    def anthropic_api_key(self):
        """Get Anthropic API key from environment."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY environment variable not set")
        return api_key

    def test_query_high_interest_topic(self, anthropic_api_key):
        """Test Claude interest estimation with a high-interest topic."""
        score = query_claude_interest(
            topic=HIGH_INTEREST_TOPIC, api_key=anthropic_api_key
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\n{HIGH_INTEREST_TOPIC} - Interest: {score:.3f}")

    def test_query_low_interest_topic(self, anthropic_api_key):
        """Test Claude interest estimation with a low-interest topic."""
        score = query_claude_interest(
            topic=LOW_INTEREST_TOPIC, api_key=anthropic_api_key
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\n{LOW_INTEREST_TOPIC} - Interest: {score:.3f}")


class TestOpenAIDivisiveness:
    """Test OpenAI divisiveness estimation."""

    @pytest.fixture
    def openai_api_key(self):
        """Get OpenAI API key from environment."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
        return api_key

    def test_query_high_divisive_topic(self, openai_api_key):
        """Test OpenAI divisiveness estimation with a divisive topic."""
        score = query_openai_divisiveness(
            topic=HIGH_DIVISIVE_TOPIC, api_key=openai_api_key
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\n{HIGH_DIVISIVE_TOPIC} - Divisiveness: {score:.3f}")

    def test_query_low_divisive_topic(self, openai_api_key):
        """Test OpenAI divisiveness estimation with a non-divisive topic."""
        score = query_openai_divisiveness(
            topic=LOW_DIVISIVE_TOPIC, api_key=openai_api_key
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\n{LOW_DIVISIVE_TOPIC} - Divisiveness: {score:.3f}")


class TestClaudeDivisiveness:
    """Test Claude divisiveness estimation."""

    @pytest.fixture
    def anthropic_api_key(self):
        """Get Anthropic API key from environment."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY environment variable not set")
        return api_key

    def test_query_high_divisive_topic(self, anthropic_api_key):
        """Test Claude divisiveness estimation with a divisive topic."""
        score = query_claude_divisiveness(
            topic=HIGH_DIVISIVE_TOPIC, api_key=anthropic_api_key
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\n{HIGH_DIVISIVE_TOPIC} - Divisiveness: {score:.3f}")

    def test_query_low_divisive_topic(self, anthropic_api_key):
        """Test Claude divisiveness estimation with a non-divisive topic."""
        score = query_claude_divisiveness(
            topic=LOW_DIVISIVE_TOPIC, api_key=anthropic_api_key
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\n{LOW_DIVISIVE_TOPIC} - Divisiveness: {score:.3f}")


class TestBuzzScore:
    """Test the unified buzz score function."""

    @pytest.fixture
    def openai_api_key(self):
        """Get OpenAI API key from environment."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
        return api_key

    @pytest.fixture
    def anthropic_api_key(self):
        """Get Anthropic API key from environment."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY environment variable not set")
        return api_key

    def test_buzz_score_openai(self, openai_api_key):
        """Test unified buzz score with OpenAI."""
        result = get_buzz_score(
            topic=HIGH_DIVISIVE_TOPIC, provider="openai", api_key=openai_api_key
        )

        assert isinstance(result, dict)
        assert "topic" in result
        assert "interest" in result
        assert "divisiveness" in result
        assert "buzz" in result
        assert "provider" in result
        assert "model" in result

        assert result["topic"] == HIGH_DIVISIVE_TOPIC
        assert result["provider"] == "openai"
        assert 0.0 <= result["interest"] <= 1.0
        assert 0.0 <= result["divisiveness"] <= 1.0
        assert 0.0 <= result["buzz"] <= 1.0

        # Verify buzz is calculated correctly
        expected_buzz = result["interest"] * result["divisiveness"]
        assert abs(result["buzz"] - expected_buzz) < 0.001

        print(f"\n{HIGH_DIVISIVE_TOPIC} (OpenAI):")
        print(f"  Interest: {result['interest']:.3f}")
        print(f"  Divisiveness: {result['divisiveness']:.3f}")
        print(f"  Buzz: {result['buzz']:.3f}")

    def test_buzz_score_claude(self, anthropic_api_key):
        """Test unified buzz score with Claude."""
        result = get_buzz_score(
            topic=HIGH_DIVISIVE_TOPIC, provider="claude", api_key=anthropic_api_key
        )

        assert isinstance(result, dict)
        assert result["topic"] == HIGH_DIVISIVE_TOPIC
        assert result["provider"] == "claude"
        assert 0.0 <= result["interest"] <= 1.0
        assert 0.0 <= result["divisiveness"] <= 1.0
        assert 0.0 <= result["buzz"] <= 1.0

        # Verify buzz is calculated correctly
        expected_buzz = result["interest"] * result["divisiveness"]
        assert abs(result["buzz"] - expected_buzz) < 0.001

        print(f"\n{HIGH_DIVISIVE_TOPIC} (Claude):")
        print(f"  Interest: {result['interest']:.3f}")
        print(f"  Divisiveness: {result['divisiveness']:.3f}")
        print(f"  Buzz: {result['buzz']:.3f}")

    def test_buzz_score_invalid_provider(self, openai_api_key):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_buzz_score(
                topic=HIGH_DIVISIVE_TOPIC, provider="invalid", api_key=openai_api_key
            )
        assert "must be either 'openai' or 'claude'" in str(exc_info.value)


class TestBuzzComparison:
    """Compare buzz scores between different topics."""

    @pytest.fixture
    def both_api_keys(self):
        """Get both API keys from environment."""
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        if not openai_key or not anthropic_key:
            pytest.skip("Both OPENAI_API_KEY and ANTHROPIC_API_KEY must be set")

        return openai_key, anthropic_key

    def test_compare_multiple_topics(self, both_api_keys):
        """Test and compare buzz scores across multiple topics."""
        openai_key, anthropic_key = both_api_keys

        topics = [
            HIGH_INTEREST_TOPIC,
            LOW_INTEREST_TOPIC,
            HIGH_DIVISIVE_TOPIC,
            LOW_DIVISIVE_TOPIC,
        ]

        print("\n" + "=" * 80)
        print("BUZZ SCORE COMPARISON")
        print("=" * 80)

        for topic in topics:
            # Get OpenAI score
            openai_result = get_buzz_score(
                topic=topic, provider="openai", api_key=openai_key
            )

            # Get Claude score
            claude_result = get_buzz_score(
                topic=topic, provider="claude", api_key=anthropic_key
            )

            print(f"\nTopic: {topic}")
            print(
                f"  OpenAI  - Interest: {openai_result['interest']:.3f}, "
                f"Divisiveness: {openai_result['divisiveness']:.3f}, "
                f"Buzz: {openai_result['buzz']:.3f}"
            )
            print(
                f"  Claude  - Interest: {claude_result['interest']:.3f}, "
                f"Divisiveness: {claude_result['divisiveness']:.3f}, "
                f"Buzz: {claude_result['buzz']:.3f}"
            )

            # Both should be valid scores
            assert 0.0 <= openai_result["buzz"] <= 1.0
            assert 0.0 <= claude_result["buzz"] <= 1.0

        print("=" * 80)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
