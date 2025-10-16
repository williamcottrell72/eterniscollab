"""
Test script for the market pipeline.

This script runs a minimal version of the pipeline to verify all components work together.
"""

import sys

sys.path.append("..")

from market_pipeline import run_market_pipeline


def main():
    print("=" * 80)
    print("MARKET PIPELINE - TEST RUN")
    print("=" * 80)
    print()
    print("This test will:")
    print("  1. Generate 2 topics with 1 question each")
    print("  2. Estimate daily volumes using 10 historical examples")
    print("  3. Estimate probabilities using 5 samples")
    print("  4. Allocate $1000 total capital")
    print("  5. Save results to data/pipelines/<timestamp>")
    print()
    print("Note: This requires OPENROUTER_API_KEY to be set")
    print("=" * 80)
    print()

    input("Press Enter to continue or Ctrl+C to cancel...")

    try:
        results = run_market_pipeline(
            n_topics=2,
            k_questions_per_topic=1,
            total_capital=1000.0,
            n_volume_examples=10,
            n_probability_samples=5,
            question_model="openai/gpt-4o-mini:online",
            volume_model="openai/gpt-4o-mini",
            probability_model="openai/gpt-4o-mini",
        )

        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nGenerated {len(results['questions'])} questions")
        print(f"Results saved to: {results['output_dir']}")

        print("\nQuestions generated:")
        for i, q in enumerate(results["questions"], 1):
            print(f"\n{i}. {q['question']}")
            print(f"   Topic: {q['topic']}")
            print(f"   Estimated Volume: ${q.get('estimated_daily_volume', 'N/A')}")
            print(f"   Probability: {q.get('probability_mean', 'N/A')}")
            print(f"   Allocated Capital: ${q.get('allocated_capital', 0):.2f}")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
