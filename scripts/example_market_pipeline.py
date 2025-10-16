"""
Example usage of the market pipeline.

This script demonstrates different ways to use the market pipeline.
"""

import sys

sys.path.append("..")

from market_pipeline import run_market_pipeline

# Example 1: Quick test run with minimal questions
print("=" * 80)
print("EXAMPLE 1: Quick Test Run")
print("=" * 80)
print("Generating 2 topics with 1 question each")
print("Capital: $1,000")
print()

results = run_market_pipeline(
    n_topics=2,
    k_questions_per_topic=1,
    total_capital=1000.0,
    n_volume_examples=10,
    n_probability_samples=5,
)

print(f"\nGenerated {len(results['questions'])} questions")
print(f"Results saved to: {results['output_dir']}")

# Example 2: Standard production run
print("\n\n" + "=" * 80)
print("EXAMPLE 2: Standard Production Run")
print("=" * 80)
print("Generating 5 topics with 2 questions each")
print("Capital: $10,000")
print()

results = run_market_pipeline(
    n_topics=5,
    k_questions_per_topic=2,
    total_capital=10000.0,
    n_volume_examples=20,
    n_probability_samples=10,
)

print(f"\nGenerated {len(results['questions'])} questions")
print(f"Results saved to: {results['output_dir']}")

# Print summary of allocations
print("\nCapital Allocations:")
for i, q in enumerate(results["questions"], 1):
    print(f"{i}. ${q['allocated_capital']:.2f} - {q['question'][:60]}...")

# Example 3: Using higher-quality models
print("\n\n" + "=" * 80)
print("EXAMPLE 3: Premium Models")
print("=" * 80)
print("Using GPT-4o for questions and Claude Sonnet for volumes")
print()

results = run_market_pipeline(
    n_topics=3,
    k_questions_per_topic=2,
    total_capital=10000.0,
    question_model="openai/gpt-4o:online",
    volume_model="anthropic/claude-sonnet-4",
    probability_model="openai/gpt-4o",
    n_volume_examples=30,
    n_probability_samples=15,
)

print(f"\nGenerated {len(results['questions'])} questions")
print(f"Results saved to: {results['output_dir']}")

print("\n" + "=" * 80)
print("All examples complete!")
print("=" * 80)
