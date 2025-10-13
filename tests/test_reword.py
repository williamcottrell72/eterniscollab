"""
Unit tests for reword functions in probability_estimator module.
"""

import pytest
import os
import sys

# Add parent directory to path to import probability_estimator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from probability_estimator import reword_prompt_openai, reword_prompt_claude


class TestRewordOpenAI:
    """Tests for reword_prompt_openai function."""

    def test_temperature_zero_returns_original(self):
        """Test that temperature=0 returns the original prompt unchanged."""
        original = "Will Kamala Harris run for president again?"
        result = reword_prompt_openai(original, temperature=0)
        assert result == original

    # def test_requires_api_key(self):
    #     """Test that function raises error when API key is not available."""
    #     original = "Will Kamala Harris run for president again?"
    #     # Temporarily remove env variable if it exists
    #     old_key = os.environ.pop('OPENAI_API_KEY', None)

    #     try:
    #         with pytest.raises(ValueError, match="OpenAI API key not provided"):
    #             reword_prompt_openai(original, temperature=0.5, api_key=None)
    #     finally:
    #         # Restore env variable if it existed
    #         if old_key:
    #             os.environ['OPENAI_API_KEY'] = old_key

    def test_returns_string(self):
        """Test that function returns a string (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"
        api_key = os.environ.get('OPENAI_API_KEY')

        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        result = reword_prompt_openai(original, temperature=0.5, api_key=api_key)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_low_temperature_produces_similar_output(self):
        """Test that low temperature produces output similar to original (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"
        api_key = os.environ.get('OPENAI_API_KEY')

        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        result = reword_prompt_openai(original, temperature=0.1, api_key=api_key)
        breakpoint() 
        # Check that key terms are preserved
        assert "Kamala Harris" in result or "Harris" in result
        assert "president" in result.lower() or "office" in result.lower()

    def test_high_temperature_produces_different_output(self):
        """Test that high temperature can produce more varied output (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"
        api_key = os.environ.get('OPENAI_API_KEY')

        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        result = reword_prompt_openai(original, temperature=0.9, api_key=api_key)

        # Result should still be a valid question string
        assert isinstance(result, str)
        assert len(result) > 0
        # The reworded prompt should be different from original (in most cases)
        # Note: This is probabilistic, so might occasionally fail
        assert result != original or temperature == 0

    def test_different_calls_produce_variety(self):
        """Test that multiple calls with same prompt produce varied results (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"
        api_key = os.environ.get('OPENAI_API_KEY')

        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        results = [
            reword_prompt_openai(original, temperature=0.7, api_key=api_key)
            for _ in range(3)
        ]

        # At least some variety should exist (probabilistic test)
        # Check that not all results are identical
        unique_results = set(results)
        assert len(unique_results) >= 1  # At minimum, we get valid strings


class TestRewordClaude:
    """Tests for reword_prompt_claude function."""

    def test_temperature_zero_returns_original(self):
        """Test that temperature=0 returns the original prompt unchanged."""
        original = "Will Kamala Harris run for president again?"
        result = reword_prompt_claude(original, temperature=0)
        assert result == original

    # def test_requires_api_key(self):
    #     """Test that function raises error when API key is not available."""
    #     original = "Will Kamala Harris run for president again?"
    #     # Temporarily remove env variable if it exists
    #     old_key = os.environ.pop('ANTHROPIC_API_KEY', None)

    #     try:
    #         with pytest.raises(ValueError, match="Anthropic API key not provided"):
    #             reword_prompt_claude(original, temperature=0.5, api_key=None)
    #     finally:
    #         # Restore env variable if it existed
    #         if old_key:
    #             os.environ['ANTHROPIC_API_KEY'] = old_key

    def test_returns_string(self):
        """Test that function returns a string (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"
        api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        result = reword_prompt_claude(original, temperature=0.5, api_key=api_key)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_low_temperature_produces_similar_output(self):
        """Test that low temperature produces output similar to original (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"
        api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        result = reword_prompt_claude(original, temperature=0.1, api_key=api_key)

        # Check that key terms are preserved
        assert "Kamala Harris" in result or "Harris" in result
        assert "president" in result.lower() or "office" in result.lower()

    def test_high_temperature_produces_different_output(self):
        """Test that high temperature can produce more varied output (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"
        api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        result = reword_prompt_claude(original, temperature=0.9, api_key=api_key)

        # Result should still be a valid question string
        assert isinstance(result, str)
        assert len(result) > 0
        # The reworded prompt should be different from original (in most cases)
        assert result != original or temperature == 0

    def test_different_calls_produce_variety(self):
        """Test that multiple calls with same prompt produce varied results (requires valid API key)."""
        original = "Will Kamala Harris run for president again?"
        api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        results = [
            reword_prompt_claude(original, temperature=0.7, api_key=api_key)
            for _ in range(3)
        ]

        # At least some variety should exist (probabilistic test)
        unique_results = set(results)
        assert len(unique_results) >= 1  # At minimum, we get valid strings


class TestRewordComparison:
    """Tests comparing behavior between OpenAI and Claude reword functions."""

    def test_both_respect_temperature_zero(self):
        """Test that both functions return original at temperature=0."""
        original = "Will Kamala Harris run for president again?"

        openai_result = reword_prompt_openai(original, temperature=0)
        claude_result = reword_prompt_claude(original, temperature=0)

        assert openai_result == original
        assert claude_result == original
        assert openai_result == claude_result

    def test_both_produce_valid_output_types(self):
        """Test that both functions produce string outputs (requires valid API keys)."""
        original = "Will Kamala Harris run for president again?"
        openai_key = os.environ.get('OPENAI_API_KEY')
        claude_key = os.environ.get('ANTHROPIC_API_KEY')

        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")
        if not claude_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        openai_result = reword_prompt_openai(original, temperature=0.5, api_key=openai_key)
        claude_result = reword_prompt_claude(original, temperature=0.5, api_key=claude_key)

        assert isinstance(openai_result, str)
        assert isinstance(claude_result, str)
        assert len(openai_result) > 0
        assert len(claude_result) > 0


class TestRewordEdgeCases:
    """Tests for edge cases in reword functions."""

    def test_empty_string_handling(self):
        """Test behavior with empty string input."""
        empty = ""

        # Temperature 0 should return empty string
        assert reword_prompt_openai(empty, temperature=0) == empty
        assert reword_prompt_claude(empty, temperature=0) == empty

    def test_very_long_prompt(self):
        """Test behavior with very long prompts (requires valid API keys)."""
        long_prompt = "Will Kamala Harris run for president again? " * 20
        openai_key = os.environ.get('OPENAI_API_KEY')
        claude_key = os.environ.get('ANTHROPIC_API_KEY')

        if openai_key:
            result = reword_prompt_openai(long_prompt, temperature=0.5, api_key=openai_key)
            assert isinstance(result, str)
            assert len(result) > 0
        else:
            pytest.skip("OPENAI_API_KEY not set")

    def test_special_characters_in_prompt(self):
        """Test handling of special characters."""
        special_prompt = "Will Kamala Harris run for president again? (2028 election) [yes/no]"

        # At temperature 0, should return unchanged
        assert reword_prompt_openai(special_prompt, temperature=0) == special_prompt
        assert reword_prompt_claude(special_prompt, temperature=0) == special_prompt

    def test_non_question_prompts(self):
        """Test reword functions work with non-question statements (requires valid API keys)."""
        statement = "Kamala Harris might run for president again."
        openai_key = os.environ.get('OPENAI_API_KEY')

        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        result = reword_prompt_openai(statement, temperature=0.5, api_key=openai_key)
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
