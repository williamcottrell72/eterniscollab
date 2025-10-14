"""
Example usage of topic_generator module.

This demonstrates how to use the topic generator in different scenarios.
"""

import os
from topic_generator import generate_topics_and_questions, print_topics_and_questions


def example_basic():
    """Basic usage example."""
    print("="*80)
    print("EXAMPLE 1: Basic Usage (N=2 topics, k=1 question)")
    print("="*80 + "\n")

    # Generate 2 topics with 1 question each (as per the requirement)
    result = generate_topics_and_questions(n_topics=2, k_questions=1)

    # Pretty print the results
    print_topics_and_questions(result)

    return result


def example_multiple_questions():
    """Example with multiple questions per topic."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Questions (N=3 topics, k=3 questions)")
    print("="*80 + "\n")

    # Generate 3 topics with up to 3 questions each
    result = generate_topics_and_questions(n_topics=3, k_questions=3)

    # Pretty print the results
    print_topics_and_questions(result)

    return result


def example_programmatic_access():
    """Example showing programmatic access to results."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Programmatic Access")
    print("="*80 + "\n")

    # Generate topics
    result = generate_topics_and_questions(n_topics=2, k_questions=2)

    # Access data programmatically
    print("Iterating through results:\n")
    for topic, questions in result.items():
        print(f"Topic: {topic}")
        print(f"Number of questions: {len(questions)}")
        for i, question in enumerate(questions, 1):
            print(f"  Q{i}: {question}")
        print()

    # Calculate statistics
    total_questions = sum(len(q) for q in result.values())
    avg_questions = total_questions / len(result) if result else 0

    print(f"Statistics:")
    print(f"  Total topics: {len(result)}")
    print(f"  Total questions: {total_questions}")
    print(f"  Average questions per topic: {avg_questions:.2f}")

    return result


def example_different_model():
    """Example using a different model."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Using Claude Instead of GPT-4o-mini")
    print("="*80 + "\n")

    # Use Claude model instead of default GPT-4o-mini
    result = generate_topics_and_questions(
        n_topics=2,
        k_questions=2,
        model="anthropic/claude-sonnet-4"
    )

    print_topics_and_questions(result)

    return result


def example_error_handling():
    """Example showing error handling."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Error Handling")
    print("="*80 + "\n")

    try:
        # This will fail if API key is not set
        result = generate_topics_and_questions(n_topics=2, k_questions=1)
        print("Success! Generated topics:")
        print_topics_and_questions(result)
        return result

    except ValueError as e:
        print(f"ValueError: {e}")
        print("This usually means:")
        print("  - Invalid parameters (n_topics or k_questions < 1)")
        print("  - Missing OPENROUTER_API_KEY environment variable")
        return None

    except Exception as e:
        print(f"Exception: {e}")
        print("This usually means:")
        print("  - API call failed")
        print("  - Network error")
        print("  - Invalid response from API")
        return None


def main():
    """Run all examples."""
    # Check if API key is set
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY environment variable not set!")
        print("Please set it before running this example:")
        print('  export OPENROUTER_API_KEY="your-key-here"')
        print("\nRunning error handling example to demonstrate this...\n")
        example_error_handling()
        return

    # Run examples
    print("Running Topic Generator Examples\n")

    # Example 1: Basic usage (N=2, k=1 as requested)
    example_basic()

    # Uncomment below to run more examples:

    # # Example 2: Multiple questions
    # example_multiple_questions()

    # # Example 3: Programmatic access
    # example_programmatic_access()

    # # Example 4: Different model (requires more tokens/cost)
    # example_different_model()

    # # Example 5: Error handling
    # example_error_handling()


if __name__ == "__main__":
    main()
