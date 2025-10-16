"""
Example script demonstrating OpenRouter integration with Pydantic AI.

This script shows how to use get_probability_distribution with various models
through OpenRouter's unified API.
"""

import asyncio
import os
from probability_estimator import get_probability_distribution, OPENROUTER_MODELS


async def main():
    """Run example probability distributions with different OpenRouter models."""

    # Example question
    question = "What is the probability that Kamala Harris runs for president again?"

    print("=" * 80)
    print("OpenRouter Probability Distribution Example")
    print("=" * 80)
    print(f"\nQuestion: {question}\n")

    # Check if API key is set
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("Please set it with: export OPENROUTER_API_KEY='your-api-key-here'")
        return

    print("Available OpenRouter Models:")
    for i, model in enumerate(OPENROUTER_MODELS, 1):
        print(f"  {i}. {model}")
    print()

    # Test with a few different models
    test_models = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "qwen/qwen-2.5-72b-instruct",
    ]

    for model in test_models:
        print(f"\n{'=' * 80}")
        print(f"Testing with model: {model}")
        print("=" * 80)

        try:
            # Run with fewer samples for demo
            result = await get_probability_distribution(
                prompt=question,
                model=model,
                n_samples=5,  # Use 5 samples for quick demo
                reword_temperature=0.5,
                prompt_temperature=0.7,
            )

            print(f"\nModel: {result['model']}")
            print(f"Samples: {result['n_samples']}")
            print(f"\nProbabilities:")
            for i, prob in enumerate(result["probabilities"], 1):
                print(f"  Sample {i}: {prob:.3f} ({prob*100:.1f}%)")

            # Calculate statistics
            import numpy as np

            probs = np.array(result["probabilities"])
            print(f"\nStatistics:")
            print(f"  Mean:   {np.mean(probs):.3f}")
            print(f"  Median: {np.median(probs):.3f}")
            print(f"  Std:    {np.std(probs):.3f}")
            print(f"  Min:    {np.min(probs):.3f}")
            print(f"  Max:    {np.max(probs):.3f}")

            print(f"\nReworded prompts:")
            for i, reworded in enumerate(
                result["reworded_prompts"][:3], 1
            ):  # Show first 3
                print(f"  {i}. {reworded}")
            if len(result["reworded_prompts"]) > 3:
                print(f"  ... and {len(result['reworded_prompts']) - 3} more")

        except Exception as e:
            print(f"\nERROR with {model}: {str(e)}")
            print(f"  Skipping to next model...")

    print(f"\n{'=' * 80}")
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
