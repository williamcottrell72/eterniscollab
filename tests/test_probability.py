"""
Unit tests for probability estimation functions.

Tests both Claude and OpenAI probability estimation functions with real API calls.
"""

import pytest
import os
import sys

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probability_estimator import (
    query_openai_probability,
    query_claude_probability,
    get_probability_estimate
)
from utils import extract_numeric_value


# Test prompt as specified
TEST_PROMPT = "What is the probability that Kamala Harris runs for president again?"


class TestProbabilityExtraction:
    """Test the probability extraction helper function."""

    def test_extract_percentage(self):
        """Test extracting probability from percentage format."""
        assert extract_numeric_value("The probability is 25%", "probability") == 0.25
        assert extract_numeric_value("I estimate 75.5% chance", "probability") == 0.755

    def test_extract_decimal(self):
        """Test extracting probability from decimal format."""
        assert extract_numeric_value("Probability: 0.25", "probability") == 0.25
        assert extract_numeric_value("The answer is 0.75", "probability") == 0.75

    def test_extract_fraction(self):
        """Test extracting probability from fraction format."""
        assert extract_numeric_value("1/4 chance", "probability") == 0.25
        assert extract_numeric_value("3/4 probability", "probability") == 0.75

    def test_invalid_text(self):
        """Test that invalid text raises ValueError."""
        with pytest.raises(ValueError):
            extract_numeric_value("No probability here", "probability")


class TestOpenAIProbability:
    """Test OpenAI probability estimation function."""

    @pytest.fixture
    def openai_api_key(self):
        """Get OpenAI API key from environment."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
        return api_key

    def test_query_openai_with_kamala_prompt(self, openai_api_key):
        """Test OpenAI probability estimation with the Kamala Harris prompt."""
        probability = query_openai_probability(
            prompt=TEST_PROMPT,
            api_key=openai_api_key,
            model="gpt-4"
        )

        # Assert the probability is a valid float between 0 and 1
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0

        # Print result for manual verification
        print(f"\nOpenAI Probability Result: {probability:.2%} ({probability:.3f})")

    def test_query_openai_with_gpt4_turbo(self, openai_api_key):
        """Test OpenAI with GPT-4 Turbo model."""
        probability = query_openai_probability(
            prompt=TEST_PROMPT,
            api_key=openai_api_key,
            model="gpt-4-turbo-preview"
        )

        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
        print(f"\nOpenAI (GPT-4 Turbo) Result: {probability:.2%} ({probability:.3f})")

    def test_query_openai_invalid_api_key(self):
        """Test that invalid API key raises exception."""
        with pytest.raises((Exception, ImportError)) as exc_info:
            query_openai_probability(
                prompt=TEST_PROMPT,
                api_key="invalid_key",
                model="gpt-4"
            )
        # Accept either API error or ImportError if openai package not installed
        assert ("API call failed" in str(exc_info.value) or
                "not installed" in str(exc_info.value))


class TestClaudeProbability:
    """Test Claude probability estimation function."""

    @pytest.fixture
    def anthropic_api_key(self):
        """Get Anthropic API key from environment."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY environment variable not set")
        return api_key

    def test_query_claude_with_kamala_prompt(self, anthropic_api_key):
        """Test Claude probability estimation with the Kamala Harris prompt."""
        probability = query_claude_probability(
            prompt=TEST_PROMPT,
            api_key=anthropic_api_key,
            model="claude-sonnet-4-5-20250929"
        )

        # Assert the probability is a valid float between 0 and 1
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0

        # Print result for manual verification
        print(f"\nClaude Probability Result: {probability:.2%} ({probability:.3f})")

    def test_query_claude_with_haiku(self, anthropic_api_key):
        """Test Claude with Haiku model."""
        probability = query_claude_probability(
            prompt=TEST_PROMPT,
            api_key=anthropic_api_key,
            model="claude-3-5-haiku-20241022"
        )

        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
        print(f"\nClaude (Haiku) Result: {probability:.2%} ({probability:.3f})")

    def test_query_claude_invalid_api_key(self):
        """Test that invalid API key raises exception."""
        with pytest.raises(Exception) as exc_info:
            query_claude_probability(
                prompt=TEST_PROMPT,
                api_key="invalid_key",
                model="claude-sonnet-4-5-20250929"
            )
        assert "API call failed" in str(exc_info.value)


class TestUnifiedInterface:
    """Test the unified get_probability_estimate function."""

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

    def test_unified_openai(self, openai_api_key):
        """Test unified interface with OpenAI provider."""
        result = get_probability_estimate(
            prompt=TEST_PROMPT,
            provider="openai",
            api_key=openai_api_key
        )

        assert isinstance(result, dict)
        assert "probability" in result
        assert "provider" in result
        assert "model" in result
        assert result["provider"] == "openai"
        assert 0.0 <= result["probability"] <= 1.0

        print(f"\nUnified OpenAI Result: {result}")

    def test_unified_claude(self, anthropic_api_key):
        """Test unified interface with Claude provider."""
        result = get_probability_estimate(
            prompt=TEST_PROMPT,
            provider="claude",
            api_key=anthropic_api_key
        )

        assert isinstance(result, dict)
        assert "probability" in result
        assert "provider" in result
        assert "model" in result
        assert result["provider"] == "claude"
        assert 0.0 <= result["probability"] <= 1.0

        print(f"\nUnified Claude Result: {result}")

    def test_unified_invalid_provider(self, openai_api_key):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_probability_estimate(
                prompt=TEST_PROMPT,
                provider="invalid",
                api_key=openai_api_key
            )
        assert "must be either 'openai' or 'claude'" in str(exc_info.value)

    def test_unified_missing_api_key(self):
        """Test that missing API key is handled properly."""
        # Temporarily clear environment variable
        old_key = os.environ.get("OPENAI_API_KEY")
        if old_key:
            del os.environ["OPENAI_API_KEY"]

        try:
            with pytest.raises(ValueError) as exc_info:
                get_probability_estimate(
                    prompt=TEST_PROMPT,
                    provider="openai",
                    api_key=None
                )
            assert "API key not provided" in str(exc_info.value)
        finally:
            # Restore environment variable
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key


class TestComparisonBetweenProviders:
    """Compare results between OpenAI and Claude."""

    @pytest.fixture
    def both_api_keys(self):
        """Get both API keys from environment."""
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        if not openai_key or not anthropic_key:
            pytest.skip("Both OPENAI_API_KEY and ANTHROPIC_API_KEY must be set for comparison tests")

        return openai_key, anthropic_key

    def test_compare_both_providers(self, both_api_keys):
        """Test and compare probability estimates from both providers."""
        openai_key, anthropic_key = both_api_keys

        # Get OpenAI estimate
        openai_result = get_probability_estimate(
            prompt=TEST_PROMPT,
            provider="openai",
            api_key=openai_key
        )

        # Get Claude estimate
        claude_result = get_probability_estimate(
            prompt=TEST_PROMPT,
            provider="claude",
            api_key=anthropic_key
        )

        # Print comparison
        print("\n" + "="*60)
        print("PROBABILITY ESTIMATE COMPARISON")
        print("="*60)
        print(f"Prompt: {TEST_PROMPT}")
        print(f"\nOpenAI ({openai_result['model']}):")
        print(f"  Probability: {openai_result['probability']:.2%} ({openai_result['probability']:.3f})")
        print(f"\nClaude ({claude_result['model']}):")
        print(f"  Probability: {claude_result['probability']:.2%} ({claude_result['probability']:.3f})")
        print(f"\nDifference: {abs(openai_result['probability'] - claude_result['probability']):.2%}")
        print("="*60)

        # Both should be valid probabilities
        assert 0.0 <= openai_result["probability"] <= 1.0
        assert 0.0 <= claude_result["probability"] <= 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
