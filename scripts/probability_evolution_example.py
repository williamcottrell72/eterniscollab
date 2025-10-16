"""
Example: Track how LLM probability estimates evolve with different knowledge cutoff dates.

This script shows a simple, practical example of using get_probability_distribution_over_time()
to understand how forecasts change as more information becomes available.
"""

import asyncio
from datetime import datetime
from probability_estimator import (
    get_probability_distribution_over_time,
    analyze_probability_evolution,
)
import numpy as np


async def main():
    """
    Example: Track how election probability estimates evolved.
    """
    print("=" * 80)
    print("Probability Evolution Example")
    print("=" * 80)
    print()
    print("Question: Will Donald Trump win the 2024 US Presidential election?")
    print()
    print("This example tracks how the model's probability estimate changes")
    print("as we vary the knowledge cutoff date from January 2024 to Election Day.")
    print()

    # Query probability distributions with different knowledge cutoff dates
    result = await get_probability_distribution_over_time(
        prompt="Will Donald Trump win the 2024 US Presidential election?",
        start_date=datetime(2024, 1, 1),  # Start of 2024
        end_date=datetime(2024, 11, 5),  # Election day
        frequency_days=30,  # Monthly updates
        n_samples=10,  # 10 samples per distribution
        model="openai/gpt-4o-mini",
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Display results
    for date_str, dist in sorted(result.items()):
        probs = dist["probabilities"]
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)
        cutoff = dist["knowledge_cutoff_date"]

        print(f"Knowledge Cutoff: {cutoff}")
        print(f"  Mean Probability: {mean_prob:.1%}")
        print(f"  Std Dev: {std_prob:.1%}")
        print(f"  Range: {min(probs):.1%} - {max(probs):.1%}")
        print()

    # Analyze the evolution
    stats = await analyze_probability_evolution(result)

    print("=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print()
    print(stats[["date", "mean", "std", "min", "max"]].to_string(index=False))

    # Calculate change over time
    first_mean = stats.iloc[0]["mean"]
    last_mean = stats.iloc[-1]["mean"]
    change = last_mean - first_mean

    print()
    print("=" * 80)
    print("CHANGE OVER TIME")
    print("=" * 80)
    print(
        f"First estimate ({stats.iloc[0]['date'].strftime('%Y-%m-%d')}): {first_mean:.1%}"
    )
    print(
        f"Last estimate ({stats.iloc[-1]['date'].strftime('%Y-%m-%d')}): {last_mean:.1%}"
    )
    print(f"Change: {change:+.1%}")

    if abs(change) > 0.1:
        print(
            "\nSignificant change detected! Forecasts evolved substantially over this period."
        )
    else:
        print("\nForecasts remained relatively stable over this period.")


if __name__ == "__main__":
    asyncio.run(main())
