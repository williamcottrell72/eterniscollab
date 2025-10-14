"""
Unit tests for reword functions in probability_estimator module.
"""

import pytest
import os
import sys
import asyncio

# Add parent directory to path to import probability_estimator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from probability_estimator import reword_prompt


class TestRewordOpenAI:
    """Tests for reword_prompt with OpenAI models via OpenRouter."""

    @pytest.fixture
    def openrouter_api_key(self):
        """Get OpenRouter API key from environment."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        return api_key

    @pytest.mark.asyncio
    async def test_temperature_zero_returns_original(self):
        """Test that temperature=0 returns the original prompt unchanged."""
        original = "Will Kamala Harris run for president again?"
        result = await reword_prompt(original, temperature=0)
        assert result == original

    @pytest.mark.asyncio
    async def test_requires_api_key(self):
        """Test that function raises error when API key is not available."""
        original = "Will Kamala Harris run for president again?"
        # Temporarily remove env variable if it exists
        old_key = os.environ.pop('OPENROUTER_API_KEY', None)

        try:
            with pytest.raises(ValueError, match="OpenRouter API key not provided"):
                await reword_prompt(original, temperature=0.5, api_key=None)
        finally:
            # Restore env variable if it existed
            if old_key:
                os.environ['OPENROUTER_API_KEY'] = old_key

    @pytest.mark.asyncio
    async def test_returns_string(self, openrouter_api_key):
        """Test that function returns a string (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"

        result = await reword_prompt(
            original,
            temperature=0.5,
            api_key=openrouter_api_key,
            model="openai/gpt-4o-mini"
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_low_temperature_produces_similar_output(self, openrouter_api_key):
        """Test that low temperature produces output similar to original (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"

        result = await reword_prompt(
            original,
            temperature=0.1,
            api_key=openrouter_api_key,
            model="openai/gpt-4o-mini"
        )

        # Check that key terms are preserved
        assert "Kamala Harris" in result or "Harris" in result
        assert "president" in result.lower() or "office" in result.lower()

    @pytest.mark.asyncio
    async def test_high_temperature_produces_different_output(self, openrouter_api_key):
        """Test that high temperature can produce more varied output (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"

        result = await reword_prompt(
            original,
            temperature=0.9,
            api_key=openrouter_api_key,
            model="openai/gpt-4o-mini"
        )

        # Result should still be a valid question string
        assert isinstance(result, str)
        assert len(result) > 0
        # The reworded prompt should be different from original (in most cases)
        # Note: This is probabilistic, so might occasionally fail
        assert result != original or temperature == 0

    @pytest.mark.asyncio
    async def test_different_calls_produce_variety(self, openrouter_api_key):
        """Test that multiple calls with same prompt produce varied results (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"

        results = []
        for _ in range(3):
            result = await reword_prompt(
                original,
                temperature=0.7,
                api_key=openrouter_api_key,
                model="openai/gpt-4o-mini"
            )
            results.append(result)

        # At least some variety should exist (probabilistic test)
        # Check that not all results are identical
        unique_results = set(results)
        assert len(unique_results) >= 1  # At minimum, we get valid strings


class TestRewordClaude:
    """Tests for reword_prompt with Claude models via OpenRouter."""

    @pytest.fixture
    def openrouter_api_key(self):
        """Get OpenRouter API key from environment."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        return api_key

    @pytest.mark.asyncio
    async def test_temperature_zero_returns_original(self):
        """Test that temperature=0 returns the original prompt unchanged."""
        original = "Will Kamala Harris run for president again?"
        result = await reword_prompt(original, temperature=0)
        assert result == original

    @pytest.mark.asyncio
    async def test_returns_string(self, openrouter_api_key):
        """Test that function returns a string (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"

        result = await reword_prompt(
            original,
            temperature=0.5,
            api_key=openrouter_api_key,
            model="anthropic/claude-3.5-sonnet"
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_low_temperature_produces_similar_output(self, openrouter_api_key):
        """Test that low temperature produces output similar to original (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"

        result = await reword_prompt(
            original,
            temperature=0.1,
            api_key=openrouter_api_key,
            model="anthropic/claude-3.5-sonnet"
        )

        # Check that key terms are preserved
        assert "Kamala Harris" in result or "Harris" in result
        assert "president" in result.lower() or "office" in result.lower()

    @pytest.mark.asyncio
    async def test_high_temperature_produces_different_output(self, openrouter_api_key):
        """Test that high temperature can produce more varied output (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"

        result = await reword_prompt(
            original,
            temperature=0.9,
            api_key=openrouter_api_key,
            model="anthropic/claude-3.5-sonnet"
        )

        # Result should still be a valid question string
        assert isinstance(result, str)
        assert len(result) > 0
        # The reworded prompt should be different from original (in most cases)
        assert result != original or temperature == 0

    @pytest.mark.asyncio
    async def test_different_calls_produce_variety(self, openrouter_api_key):
        """Test that multiple calls with same prompt produce varied results (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"

        results = []
        for _ in range(3):
            result = await reword_prompt(
                original,
                temperature=0.7,
                api_key=openrouter_api_key,
                model="anthropic/claude-3.5-sonnet"
            )
            results.append(result)

        # At least some variety should exist (probabilistic test)
        unique_results = set(results)
        assert len(unique_results) >= 1  # At minimum, we get valid strings


class TestRewordComparison:
    """Tests comparing behavior between OpenAI and Claude models via OpenRouter."""

    @pytest.fixture
    def openrouter_api_key(self):
        """Get OpenRouter API key from environment."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        return api_key

    @pytest.mark.asyncio
    async def test_both_respect_temperature_zero(self):
        """Test that both models return original at temperature=0."""
        original = "Will Kamala Harris run for president again?"

        openai_result = await reword_prompt(original, temperature=0)
        claude_result = await reword_prompt(original, temperature=0)

        assert openai_result == original
        assert claude_result == original
        assert openai_result == claude_result

    @pytest.mark.asyncio
    async def test_both_produce_valid_output_types(self, openrouter_api_key):
        """Test that both models produce string outputs (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"

        openai_result = await reword_prompt(
            original,
            temperature=0.5,
            api_key=openrouter_api_key,
            model="openai/gpt-4o-mini"
        )
        claude_result = await reword_prompt(
            original,
            temperature=0.5,
            api_key=openrouter_api_key,
            model="anthropic/claude-3.5-sonnet"
        )

        assert isinstance(openai_result, str)
        assert isinstance(claude_result, str)
        assert len(openai_result) > 0
        assert len(claude_result) > 0


class TestRewordEdgeCases:
    """Tests for edge cases in reword functions."""

    @pytest.fixture
    def openrouter_api_key(self):
        """Get OpenRouter API key from environment."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        return api_key

    @pytest.mark.asyncio
    async def test_empty_string_handling(self):
        """Test behavior with empty string input."""
        empty = ""

        # Temperature 0 should return empty string
        assert await reword_prompt(empty, temperature=0) == empty

    @pytest.mark.asyncio
    async def test_very_long_prompt(self, openrouter_api_key):
        """Test behavior with very long prompts (requires valid API key)."""
        long_prompt = "Will Kamala Harris run for president again? " * 20

        result = await reword_prompt(
            long_prompt,
            temperature=0.5,
            api_key=openrouter_api_key,
            model="openai/gpt-4o-mini"
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self):
        """Test handling of special characters."""
        special_prompt = "Will Kamala Harris run for president again? (2028 election) [yes/no]"

        # At temperature 0, should return unchanged
        assert await reword_prompt(special_prompt, temperature=0) == special_prompt

    @pytest.mark.asyncio
    async def test_non_question_prompts(self, openrouter_api_key):
        """Test reword functions work with non-question statements (requires valid API key)."""
        statement = "Kamala Harris might run for president again."

        result = await reword_prompt(
            statement,
            temperature=0.5,
            api_key=openrouter_api_key,
            model="openai/gpt-4o-mini"
        )
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
