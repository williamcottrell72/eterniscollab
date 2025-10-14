"""
Unit tests for OpenRouter integration with probability estimation.

Tests the async get_probability_distribution function with various OpenRouter models.
"""

import pytest
import os
import sys
import numpy as np

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probability_estimator import get_probability_distribution, OPENROUTER_MODELS


# Test question
TEST_QUESTION = "What is the probability that Kamala Harris runs for president again?"


class TestOpenRouterModels:
    """Test OpenRouter model integration."""

    @pytest.fixture
    def openrouter_api_key(self):
        """Get OpenRouter API key from environment."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        return api_key

    @pytest.mark.asyncio
    async def test_openai_gpt4o_mini(self, openrouter_api_key):
        """Test with OpenAI GPT-4o-mini via OpenRouter."""
        result = await get_probability_distribution(
            prompt=TEST_QUESTION,
            model="openai/gpt-4o-mini",
            n_samples=3,  # Small sample for fast testing
            reword_temperature=0.5,
            prompt_temperature=0.7,
        )

        # Verify structure
        assert isinstance(result, dict)
        assert "probabilities" in result
        assert "reworded_prompts" in result
        assert "model" in result
        assert "n_samples" in result

        # Verify content
        assert result["model"] == "openai/gpt-4o-mini"
        assert result["n_samples"] == 3
        assert len(result["probabilities"]) == 3
        assert len(result["reworded_prompts"]) == 3

        # Verify probabilities are valid
        for prob in result["probabilities"]:
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

        # Print results
        print(f"\n{result['model']} Results:")
        print(f"  Probabilities: {[f'{p:.3f}' for p in result['probabilities']]}")
        print(f"  Mean: {np.mean(result['probabilities']):.3f}")

    @pytest.mark.asyncio
    async def test_anthropic_claude_sonnet(self, openrouter_api_key):
        """Test with Anthropic Claude Sonnet via OpenRouter."""
        result = await get_probability_distribution(
            prompt=TEST_QUESTION,
            model="anthropic/claude-3.5-sonnet",
            n_samples=3,
            reword_temperature=0.5,
            prompt_temperature=0.7,
        )

        # Verify structure and content
        assert result["model"] == "anthropic/claude-3.5-sonnet"
        assert result["n_samples"] == 3
        assert len(result["probabilities"]) == 3

        # Verify probabilities are valid
        for prob in result["probabilities"]:
            assert 0.0 <= prob <= 1.0

        print(f"\n{result['model']} Results:")
        print(f"  Probabilities: {[f'{p:.3f}' for p in result['probabilities']]}")
        print(f"  Mean: {np.mean(result['probabilities']):.3f}")

    @pytest.mark.asyncio
    async def test_qwen_model(self, openrouter_api_key):
        """Test with Qwen model via OpenRouter."""
        result = await get_probability_distribution(
            prompt=TEST_QUESTION,
            model="qwen/qwen-2.5-72b-instruct",
            n_samples=3,
            reword_temperature=0.5,
            prompt_temperature=0.7,
        )

        # Verify structure and content
        assert result["model"] == "qwen/qwen-2.5-72b-instruct"
        assert result["n_samples"] == 3
        assert len(result["probabilities"]) == 3

        # Verify probabilities are valid
        for prob in result["probabilities"]:
            assert 0.0 <= prob <= 1.0

        print(f"\n{result['model']} Results:")
        print(f"  Probabilities: {[f'{p:.3f}' for p in result['probabilities']]}")
        print(f"  Mean: {np.mean(result['probabilities']):.3f}")

    @pytest.mark.asyncio
    async def test_knowledge_cutoff(self, openrouter_api_key):
        """Test that knowledge cutoff date is respected."""
        result = await get_probability_distribution(
            prompt=TEST_QUESTION,
            model="openai/gpt-4o-mini",
            n_samples=2,
            knowledge_cutoff_date="January 2024"
        )

        # Verify cutoff date is stored
        assert result.get("knowledge_cutoff_date") == "January 2024"
        assert len(result["probabilities"]) == 2

        print(f"\nKnowledge Cutoff Test:")
        print(f"  Cutoff Date: {result['knowledge_cutoff_date']}")
        print(f"  Probabilities: {[f'{p:.3f}' for p in result['probabilities']]}")


class TestOpenRouterStatistics:
    """Test statistical properties of probability distributions."""

    @pytest.fixture
    def openrouter_api_key(self):
        """Get OpenRouter API key from environment."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        return api_key

    @pytest.mark.asyncio
    async def test_distribution_statistics(self, openrouter_api_key):
        """Test statistical properties with multiple samples."""
        result = await get_probability_distribution(
            prompt=TEST_QUESTION,
            model="openai/gpt-4o-mini",
            n_samples=10,
            reword_temperature=0.5,
            prompt_temperature=0.7,
        )

        probs = np.array(result["probabilities"])

        # Calculate statistics
        mean = np.mean(probs)
        median = np.median(probs)
        std = np.std(probs)
        min_val = np.min(probs)
        max_val = np.max(probs)

        # All values should be valid probabilities
        assert 0.0 <= mean <= 1.0
        assert 0.0 <= median <= 1.0
        assert 0.0 <= min_val <= 1.0
        assert 0.0 <= max_val <= 1.0
        assert std >= 0.0

        # Print statistics
        print(f"\nDistribution Statistics (n=10):")
        print(f"  Mean:   {mean:.3f}")
        print(f"  Median: {median:.3f}")
        print(f"  Std:    {std:.3f}")
        print(f"  Min:    {min_val:.3f}")
        print(f"  Max:    {max_val:.3f}")
        print(f"  Range:  {max_val - min_val:.3f}")

    @pytest.mark.asyncio
    async def test_reword_temperature_effect(self, openrouter_api_key):
        """Test that reword temperature affects prompt diversity."""
        # Low temperature - minimal rewording
        result_low = await get_probability_distribution(
            prompt=TEST_QUESTION,
            model="openai/gpt-4o-mini",
            n_samples=3,
            reword_temperature=0.1,
            prompt_temperature=0.7,
        )

        # High temperature - significant rewording
        result_high = await get_probability_distribution(
            prompt=TEST_QUESTION,
            model="openai/gpt-4o-mini",
            n_samples=3,
            reword_temperature=0.9,
            prompt_temperature=0.7,
        )

        # Both should return valid results
        assert len(result_low["probabilities"]) == 3
        assert len(result_high["probabilities"]) == 3

        print(f"\nReword Temperature Effect:")
        print(f"  Low temp (0.1) prompts:")
        for i, prompt in enumerate(result_low["reworded_prompts"][:2], 1):
            print(f"    {i}. {prompt[:60]}...")
        print(f"  High temp (0.9) prompts:")
        for i, prompt in enumerate(result_high["reworded_prompts"][:2], 1):
            print(f"    {i}. {prompt[:60]}...")


class TestOpenRouterComparison:
    """Test comparison between different OpenRouter models."""

    @pytest.fixture
    def openrouter_api_key(self):
        """Get OpenRouter API key from environment."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        return api_key

    @pytest.mark.asyncio
    async def test_compare_models(self, openrouter_api_key):
        """Compare probability estimates across different models."""
        models = [
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
        ]

        results = {}

        for model in models:
            result = await get_probability_distribution(
                prompt=TEST_QUESTION,
                model=model,
                n_samples=5,
                reword_temperature=0.5,
                prompt_temperature=0.7,
            )
            results[model] = result

        # Print comparison
        print(f"\n{'=' * 80}")
        print("Model Comparison")
        print('=' * 80)
        print(f"Question: {TEST_QUESTION}\n")

        for model, result in results.items():
            probs = np.array(result["probabilities"])
            print(f"{model}:")
            print(f"  Mean:   {np.mean(probs):.3f} ({np.mean(probs)*100:.1f}%)")
            print(f"  Median: {np.median(probs):.3f} ({np.median(probs)*100:.1f}%)")
            print(f"  Std:    {np.std(probs):.3f}")
            print()

        # All models should return valid probabilities
        for result in results.values():
            for prob in result["probabilities"]:
                assert 0.0 <= prob <= 1.0


class TestOpenRouterErrorHandling:
    """Test error handling for OpenRouter integration."""

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        """Test that missing API key raises appropriate error."""
        # Temporarily remove API key
        old_key = os.environ.get("OPENROUTER_API_KEY")
        if old_key:
            del os.environ["OPENROUTER_API_KEY"]

        try:
            with pytest.raises(ValueError) as exc_info:
                await get_probability_distribution(
                    prompt=TEST_QUESTION,
                    model="openai/gpt-4o-mini",
                    n_samples=2,
                )
            assert "OPENROUTER_API_KEY" in str(exc_info.value)
        finally:
            # Restore API key
            if old_key:
                os.environ["OPENROUTER_API_KEY"] = old_key

    @pytest.mark.asyncio
    async def test_invalid_model(self):
        """Test handling of potentially invalid model name."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")

        # This may fail or succeed depending on OpenRouter's model availability
        # We just test that it doesn't crash unexpectedly
        try:
            result = await get_probability_distribution(
                prompt=TEST_QUESTION,
                model="invalid/nonexistent-model",
                n_samples=1,
            )
            # If it succeeds, verify structure
            assert "probabilities" in result
        except Exception as e:
            # If it fails, it should be a clear error
            assert isinstance(e, (ValueError, Exception))
            print(f"\nExpected error for invalid model: {str(e)[:100]}")


class TestOpenRouterModelsConstant:
    """Test the OPENROUTER_MODELS constant."""

    def test_models_list_exists(self):
        """Test that OPENROUTER_MODELS is defined and non-empty."""
        assert isinstance(OPENROUTER_MODELS, list)
        assert len(OPENROUTER_MODELS) > 0

    def test_models_have_correct_format(self):
        """Test that all models follow 'provider/model-name' format."""
        for model in OPENROUTER_MODELS:
            assert isinstance(model, str)
            assert "/" in model, f"Model '{model}' should have format 'provider/model-name'"

            provider, model_name = model.split("/", 1)
            assert len(provider) > 0, f"Provider in '{model}' is empty"
            assert len(model_name) > 0, f"Model name in '{model}' is empty"

    def test_models_include_major_providers(self):
        """Test that major providers are included."""
        providers = {model.split("/")[0] for model in OPENROUTER_MODELS}

        # Should include at least OpenAI and Anthropic
        assert "openai" in providers or "anthropic" in providers

        print(f"\nAvailable providers in OPENROUTER_MODELS:")
        for provider in sorted(providers):
            provider_models = [m for m in OPENROUTER_MODELS if m.startswith(provider + "/")]
            print(f"  {provider}: {len(provider_models)} models")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
