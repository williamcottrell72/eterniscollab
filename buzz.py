"""
Buzz Analyzer using OpenAI and Claude APIs

This module provides functions to quantify the "buzz" around topics by measuring:
1. Interest level (raw public attention via internet traffic, news coverage)
2. Divisiveness (how polarizing/controversial the topic is)
"""

import json
from typing import Optional, Dict, Any, List, Tuple
import os
from utils import extract_numeric_value, query_llm_for_numeric_value


# Prompt instructions for interest level estimation
INTEREST_INSTRUCTIONS = """Rate the current level of public interest in this topic on a scale from 0 to 1, where:
- 0 = Virtually no public interest, minimal to no internet traffic or news coverage
- 0.25 = Low interest, occasional mentions in niche sources
- 0.5 = Moderate interest, regular news coverage and social media discussion
- 0.75 = High interest, frequent major news coverage and trending online
- 1.0 = Extremely high interest, dominating news cycles and viral across all platforms

Consider recent internet traffic, news coverage, social media trends, and search volume.
Provide ONLY a numeric score between 0 and 1, with no additional explanation."""


# Prompt instructions for divisiveness estimation
DIVISIVENESS_INSTRUCTIONS = """Rate how divisive and polarizing this topic is on a scale from 0 to 1, where:
- 0 = Not divisive at all, broad consensus and calm discussion
- 0.25 = Slightly divisive, some disagreement but mostly civil discourse
- 0.5 = Moderately divisive, clear opposing viewpoints with passionate debate
- 0.75 = Highly divisive, strong emotions, heated arguments, and tribal camps
- 1.0 = Extremely divisive, intense anger, extreme rhetoric, and deep polarization

Consider the intensity of emotions, presence of opposing camps, anger levels, and extremity of rhetoric.
Provide ONLY a numeric score between 0 and 1, with no additional explanation."""


def _generate_interest_prompts(topic: str) -> Tuple[str, str]:
    """
    Generate system and user prompts for interest level estimation.

    Args:
        topic: The topic to evaluate

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert analyst who evaluates public interest in topics based on news coverage,
internet traffic, social media trends, and search volume. You provide objective numerical assessments."""

    user_prompt = f"Topic: {topic}\n\n{INTEREST_INSTRUCTIONS}"

    return system_prompt, user_prompt


def _generate_divisiveness_prompts(topic: str) -> Tuple[str, str]:
    """
    Generate system and user prompts for divisiveness estimation.

    Args:
        topic: The topic to evaluate

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert analyst who evaluates how divisive and polarizing topics are based on
public discourse, emotional intensity, presence of opposing camps, and rhetoric extremity. You provide objective numerical assessments."""

    user_prompt = f"Topic: {topic}\n\n{DIVISIVENESS_INSTRUCTIONS}"

    return system_prompt, user_prompt


def query_openai_interest(
    topic: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5
) -> float:
    """
    Query OpenAI API to get an interest level score for a topic.

    Args:
        topic: The topic or subject to evaluate
        api_key: OpenAI API key
        model: Model to use (default: "gpt-4o-mini")
        temperature: Sampling temperature (default: 0.5)

    Returns:
        float: Interest score between 0 and 1

    Raises:
        ValueError: If unable to extract a valid score from the response
        Exception: If API call fails
    """
    system_prompt, user_prompt = _generate_interest_prompts(topic)

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider="openai",
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=100,
        value_name="interest score"
    )


def query_claude_interest(
    topic: str,
    api_key: str,
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.5
) -> float:
    """
    Query Claude API to get an interest level score for a topic.

    Args:
        topic: The topic or subject to evaluate
        api_key: Anthropic API key
        model: Model to use (default: "claude-sonnet-4-5-20250929")
        temperature: Sampling temperature (default: 0.5)

    Returns:
        float: Interest score between 0 and 1

    Raises:
        ValueError: If unable to extract a valid score from the response
        Exception: If API call fails
    """
    system_prompt, user_prompt = _generate_interest_prompts(topic)

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider="claude",
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=100,
        value_name="interest score"
    )


def query_openai_divisiveness(
    topic: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5
) -> float:
    """
    Query OpenAI API to get a divisiveness score for a topic.

    Args:
        topic: The topic or subject to evaluate
        api_key: OpenAI API key
        model: Model to use (default: "gpt-4o-mini")
        temperature: Sampling temperature (default: 0.5)

    Returns:
        float: Divisiveness score between 0 and 1

    Raises:
        ValueError: If unable to extract a valid score from the response
        Exception: If API call fails
    """
    system_prompt, user_prompt = _generate_divisiveness_prompts(topic)

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider="openai",
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=100,
        value_name="divisiveness score"
    )


def query_claude_divisiveness(
    topic: str,
    api_key: str,
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.5
) -> float:
    """
    Query Claude API to get a divisiveness score for a topic.

    Args:
        topic: The topic or subject to evaluate
        api_key: Anthropic API key
        model: Model to use (default: "claude-sonnet-4-5-20250929")
        temperature: Sampling temperature (default: 0.5)

    Returns:
        float: Divisiveness score between 0 and 1

    Raises:
        ValueError: If unable to extract a valid score from the response
        Exception: If API call fails
    """
    system_prompt, user_prompt = _generate_divisiveness_prompts(topic)

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider="claude",
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=100,
        value_name="divisiveness score"
    )


def get_buzz_score(
    topic: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.5
) -> Dict[str, Any]:
    """
    Get both interest and divisiveness scores for a topic.

    Args:
        topic: The topic or subject to evaluate
        provider: Either "openai" or "claude" (default: "openai")
        api_key: API key (if None, will try to read from environment variables)
        model: Model to use (if None, uses default for provider)
        temperature: Sampling temperature (default: 0.5)

    Returns:
        dict: Contains:
            - 'topic': The topic evaluated
            - 'interest': Interest score (0-1)
            - 'divisiveness': Divisiveness score (0-1)
            - 'buzz': Combined buzz score (0-1), calculated as interest * divisiveness
            - 'provider': Provider used
            - 'model': Model used

    Raises:
        ValueError: If provider is invalid or API key not found
    """
    provider = provider.lower()

    if provider not in ["openai", "claude"]:
        raise ValueError("Provider must be either 'openai' or 'claude'")

    # Get API key from environment if not provided
    if api_key is None:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        else:  # claude
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")

    # Set default model
    if model is None:
        model = "gpt-4o-mini" if provider == "openai" else "claude-sonnet-4-5-20250929"

    # Query for both metrics
    if provider == "openai":
        interest = query_openai_interest(topic, api_key, model, temperature)
        divisiveness = query_openai_divisiveness(topic, api_key, model, temperature)
    else:  # claude
        interest = query_claude_interest(topic, api_key, model, temperature)
        divisiveness = query_claude_divisiveness(topic, api_key, model, temperature)

    # Calculate combined buzz score (interest * divisiveness)
    # High buzz requires both high interest AND high divisiveness
    buzz = interest * divisiveness

    return {
        "topic": topic,
        "interest": interest,
        "divisiveness": divisiveness,
        "buzz": buzz,
        "provider": provider,
        "model": model
    }


if __name__ == "__main__":
    # Example usage
    import sys

    example_topics = [
        "Artificial Intelligence",
        "Climate Change",
        "2024 US Presidential Election",
        "The Beatles",
        "Quantum Computing"
    ]

    print("Buzz Analyzer - Example Usage\n")
    print("This tool measures two aspects of 'buzz':")
    print("  1. Interest: How much public attention the topic receives")
    print("  2. Divisiveness: How polarizing and controversial the topic is")
    print("  3. Buzz: Combined score (interest × divisiveness)\n")

    # You can test by providing provider and API key as command line arguments
    # python buzz.py <provider> <api_key> [topic]

    if len(sys.argv) >= 3:
        provider = sys.argv[1]
        api_key = sys.argv[2]
        topic = sys.argv[3] if len(sys.argv) >= 4 else example_topics[2]

        try:
            print(f"Analyzing topic: {topic}\n")
            result = get_buzz_score(topic, provider=provider, api_key=api_key)

            print(f"Provider: {result['provider']}")
            print(f"Model: {result['model']}")
            print(f"\nResults:")
            print(f"  Interest:      {result['interest']:.3f} ({result['interest']*100:.1f}%)")
            print(f"  Divisiveness:  {result['divisiveness']:.3f} ({result['divisiveness']*100:.1f}%)")
            print(f"  Buzz Score:    {result['buzz']:.3f} ({result['buzz']*100:.1f}%)")

            # Interpretation
            print(f"\nInterpretation:")
            if result['buzz'] > 0.5:
                print("  🔥 HIGH BUZZ - This topic has significant attention and controversy")
            elif result['buzz'] > 0.25:
                print("  📊 MODERATE BUZZ - This topic has notable presence in discourse")
            else:
                print("  📉 LOW BUZZ - This topic has limited attention or controversy")

        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("To test, run:")
        print("  python buzz.py openai <your-openai-api-key> \"Topic Name\"")
        print("  python buzz.py claude <your-anthropic-api-key> \"Topic Name\"")
        print("\nOr set environment variables OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print(f"\nExample topics to try:")
        for topic in example_topics:
            print(f"  - {topic}")
