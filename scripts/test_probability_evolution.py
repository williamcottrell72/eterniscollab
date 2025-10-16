"""
Test script for probability distribution evolution over time.

This script demonstrates how to use get_probability_distribution_over_time()
to track how forecasts change with different knowledge cutoff dates.
"""

import asyncio
from datetime import datetime
from probability_estimator import (
    get_probability_distribution_over_time,
    analyze_probability_evolution,
)
import numpy as np


async def test_simple_evolution():
    """
    Test tracking probability evolution over a simple date range.
    """
    print("=" * 80)
    print("TEST: Simple Probability Evolution")
    print("=" * 80)
    print()

    # Example: How did forecasts of Trump winning the 2024 election
    # change from January 2024 to the election day?
    prompt = "Will Donald Trump win the 2024 US Presidential election?"

    # Query from Jan 1 to Nov 5, 2024 with monthly frequency
    result = await get_probability_distribution_over_time(
        prompt=prompt,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 11, 5),
        frequency_days=30,  # Monthly
        n_samples=5,  # Small for testing
        model="openai/gpt-4o-mini",
    )

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for date_str, dist in sorted(result.items()):
        probs = dist["probabilities"]
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)
        print(
            f"{date_str}: {mean_prob:.3f} ± {std_prob:.3f} "
            f"(range: {min(probs):.3f} - {max(probs):.3f})"
        )

    # Analyze evolution
    stats_df = await analyze_probability_evolution(result)
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    print(stats_df.to_string(index=False))

    return result, stats_df


async def test_weekly_evolution():
    """
    Test tracking weekly probability evolution over a shorter period.
    """
    print("\n\n" + "=" * 80)
    print("TEST: Weekly Probability Evolution")
    print("=" * 80)
    print()

    # Example: Track a shorter timeframe with weekly updates
    prompt = "Will the Federal Reserve raise interest rates by December 31, 2025?"

    # Query October 2024 with weekly frequency
    result = await get_probability_distribution_over_time(
        prompt=prompt,
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 31),
        frequency_days=7,  # Weekly
        n_samples=5,
        model="openai/gpt-4o-mini",
    )

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for date_str, dist in sorted(result.items()):
        probs = dist["probabilities"]
        mean_prob = np.mean(probs)
        print(f"{date_str}: {mean_prob:.3f}")

    return result


async def test_custom_parameters():
    """
    Test with custom parameters passed through kwargs.
    """
    print("\n\n" + "=" * 80)
    print("TEST: Custom Parameters")
    print("=" * 80)
    print()

    prompt = "Will SpaceX successfully land humans on Mars by 2030?"

    # Use custom parameters
    result = await get_probability_distribution_over_time(
        prompt=prompt,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
        frequency_days=7,
        n_samples=3,  # Very small for quick test
        reword_temperature=0.7,  # More variation in rewording
        prompt_temperature=0.5,  # Less variation in responses
        model="openai/gpt-4o-mini",
    )

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for date_str, dist in sorted(result.items()):
        probs = dist["probabilities"]
        print(f"{date_str}:")
        print(f"  Probabilities: {[f'{p:.3f}' for p in probs]}")
        print(f"  Mean: {np.mean(probs):.3f}")
        print(f"  Reworded prompts:")
        for i, prompt_text in enumerate(dist["reworded_prompts"], 1):
            print(f"    {i}. {prompt_text}")
        print()

    return result


async def main():
    """
    Run all tests.
    """
    print("Testing Probability Distribution Evolution Over Time")
    print("=" * 80)
    print()

    try:
        # Test 1: Simple evolution
        result1, stats1 = await test_simple_evolution()

        # Test 2: Weekly evolution
        result2 = await test_weekly_evolution()

        # Test 3: Custom parameters
        result3 = await test_custom_parameters()

        print("\n\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run async tests
    asyncio.run(main())
