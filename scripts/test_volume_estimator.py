"""
Test script for the volume estimator function in buzz.py

This script demonstrates how to use the estimate_daily_volume function
to predict average daily trading volume for new prediction market questions.
"""

from buzz import estimate_daily_volume

# Example queries to test
test_queries = [
    "Will Bitcoin exceed $100k by end of 2025?",
    "Will Donald Trump be elected president in 2028?",
    "Will it rain in San Francisco tomorrow?",
    "Will Apple's stock price reach $250 by June 2025?",
]


def main():
    print("=" * 80)
    print("DAILY VOLUME ESTIMATOR - TEST SCRIPT")
    print("=" * 80)
    print()
    print("This script estimates average daily trading volume for prediction markets")
    print("based on historical Polymarket data.\n")

    # Test with first query and print the prompt
    test_query = test_queries[0]

    print(f"Testing with query: '{test_query}'")
    print(f"Using 20 historical examples\n")

    try:
        result = estimate_daily_volume(
            query=test_query, n_examples=20, print_prompt=True  # Show the full prompt
        )

        print("\n" + "=" * 80)
        print("RESULT")
        print("=" * 80)
        print(f"\nQuery: {result['query']}")
        print(f"Estimated Daily Volume: ${result['estimated_daily_volume']:.2f}")
        print(f"Number of Examples Used: {result['n_examples']}")
        print(f"Model: {result['model']}")

        print("\n" + "=" * 80)
        print("EXAMPLE MARKETS USED (first 5)")
        print("=" * 80)
        for i, ex in enumerate(result["examples"][:5], 1):
            print(f"\n{i}. {ex['question']}")
            print(f"   Daily Volume: ${ex['daily_volume']:.2f}")

        if len(result["examples"]) > 5:
            print(f"\n... and {len(result['examples']) - 5} more examples")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. OPENROUTER_API_KEY environment variable is set")
        print("2. Closed markets data is cached (run get_all_closed_markets() first)")
        return

    print("\n" + "=" * 80)
    print("TEST ADDITIONAL QUERIES (without printing prompts)")
    print("=" * 80)

    for query in test_queries[1:]:
        try:
            result = estimate_daily_volume(
                query=query,
                n_examples=10,  # Use fewer examples for speed
                print_prompt=False,
            )
            print(f"\nQuery: {query}")
            print(f"Estimate: ${result['estimated_daily_volume']:.2f}/day")
        except Exception as e:
            print(f"\nQuery: {query}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
