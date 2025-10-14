"""
Buzz Analyzer using OpenAI and Claude APIs

This module provides functions to quantify the "buzz" around topics by measuring:
1. Interest level (raw public attention via internet traffic, news coverage)
2. Divisiveness (how polarizing/controversial the topic is)
"""

import json
from typing import Optional, Dict, Any, List, Tuple
import os
from utils import extract_numeric_value, query_llm_for_numeric_value, generate_knowledge_cutoff_instruction


def _infer_provider_from_model(model: str) -> str:
    """
    Infer the provider (openai or claude) from the model name.

    Args:
        model: Model identifier

    Returns:
        str: Either "openai" or "claude"

    Raises:
        ValueError: If provider cannot be inferred from model name
    """
    model_lower = model.lower()

    # Check for OpenAI models
    if any(prefix in model_lower for prefix in ["gpt", "o1", "o3"]):
        return "openai"

    # Check for Claude models
    if "claude" in model_lower:
        return "claude"

    raise ValueError(f"Cannot infer provider from model name '{model}'. Model must contain 'gpt', 'o1', 'o3', or 'claude'.")


# Prompt instructions for interest level estimation
INTEREST_INSTRUCTIONS = """Provide your best estimate of the current level of public interest in this topic using a continuous numeric scale from 0 to 1.

Use the full range of the scale to capture nuances:
- Near 0: Virtually no public interest (minimal internet traffic or news coverage)
- Around 0.2-0.3: Low but measurable interest (occasional niche mentions)
- Around 0.4-0.6: Moderate interest (regular news coverage and social media discussion)
- Around 0.7-0.8: High interest (frequent major news coverage, trending online)
- Near 1.0: Extremely high interest (dominating news cycles, viral across platforms)

Consider recent internet traffic, news coverage, social media trends, and search volume. Use all available information to provide your most accurate continuous estimate.

Provide ONLY a numeric score between 0 and 1 (e.g., 0.73, 0.42, 0.88), with no additional explanation."""


# Prompt instructions for divisiveness estimation
DIVISIVENESS_INSTRUCTIONS = """Provide your best estimate of how divisive and polarizing this topic is using a continuous numeric scale from 0 to 1.

Use the full range of the scale to capture nuances:
- Near 0: Not divisive at all (broad consensus, calm discussion)
- Around 0.2-0.3: Slightly divisive (some disagreement, mostly civil)
- Around 0.4-0.6: Moderately divisive (clear opposing viewpoints, passionate debate)
- Around 0.7-0.8: Highly divisive (strong emotions, heated arguments, tribal camps)
- Near 1.0: Extremely divisive (intense anger, extreme rhetoric, deep polarization)

Consider the intensity of emotions, presence of opposing camps, anger levels, and extremity of rhetoric. Use all available information to provide your most accurate continuous estimate.

Provide ONLY a numeric score between 0 and 1 (e.g., 0.67, 0.91, 0.35), with no additional explanation."""


def _generate_interest_prompts(topic: str, knowledge_cutoff_date: Optional[str] = None) -> Tuple[str, str]:
    """
    Generate system and user prompts for interest level estimation.

    Args:
        topic: The topic to evaluate
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert analyst who evaluates public interest in topics based on news coverage,
internet traffic, social media trends, and search volume. You provide objective numerical assessments."""

    # Add knowledge cutoff instruction if provided
    system_prompt += generate_knowledge_cutoff_instruction(knowledge_cutoff_date)

    user_prompt = f"Topic: {topic}\n\n{INTEREST_INSTRUCTIONS}"

    return system_prompt, user_prompt


def _generate_divisiveness_prompts(topic: str, knowledge_cutoff_date: Optional[str] = None) -> Tuple[str, str]:
    """
    Generate system and user prompts for divisiveness estimation.

    Args:
        topic: The topic to evaluate
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert analyst who evaluates how divisive and polarizing topics are based on
public discourse, emotional intensity, presence of opposing camps, and rhetoric extremity. You provide objective numerical assessments."""

    # Add knowledge cutoff instruction if provided
    system_prompt += generate_knowledge_cutoff_instruction(knowledge_cutoff_date)

    user_prompt = f"Topic: {topic}\n\n{DIVISIVENESS_INSTRUCTIONS}"

    return system_prompt, user_prompt


def query_interest(
    topic: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None
) -> float:
    """
    Query LLM API to get an interest level score for a topic.

    Provider is automatically inferred from model name:
    - Models containing 'gpt', 'o1', or 'o3' use OpenAI
    - Models containing 'claude' use Anthropic

    Args:
        topic: The topic or subject to evaluate
        api_key: API key for the inferred provider
        model: Model to use (default: "gpt-4o-mini")
        temperature: Sampling temperature (default: 0.5)
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        float: Interest score between 0 and 1

    Raises:
        ValueError: If unable to extract a valid score from the response or cannot infer provider
        Exception: If API call fails
    """
    provider = _infer_provider_from_model(model)
    system_prompt, user_prompt = _generate_interest_prompts(topic, knowledge_cutoff_date)

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=100,
        value_name="interest score"
    )


def query_divisiveness(
    topic: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None
) -> float:
    """
    Query LLM API to get a divisiveness score for a topic.

    Provider is automatically inferred from model name:
    - Models containing 'gpt', 'o1', or 'o3' use OpenAI
    - Models containing 'claude' use Anthropic

    Args:
        topic: The topic or subject to evaluate
        api_key: API key for the inferred provider
        model: Model to use (default: "gpt-4o-mini")
        temperature: Sampling temperature (default: 0.5)
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        float: Divisiveness score between 0 and 1

    Raises:
        ValueError: If unable to extract a valid score from the response or cannot infer provider
        Exception: If API call fails
    """
    provider = _infer_provider_from_model(model)
    system_prompt, user_prompt = _generate_divisiveness_prompts(topic, knowledge_cutoff_date)

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=100,
        value_name="divisiveness score"
    )


def query_interest_openrouter(
    topic: str,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None
) -> float:
    """
    Query LLM via OpenRouter API to get an interest level score for a topic.

    Args:
        topic: The topic or subject to evaluate
        model: OpenRouter model identifier (default: "openai/gpt-4o-mini")
                Format: "provider/model-name" (e.g., "anthropic/claude-sonnet-4")
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY environment variable)
        temperature: Sampling temperature (default: 0.5)
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        float: Interest score between 0 and 1

    Raises:
        ValueError: If unable to extract a valid score from the response or API key not found
        Exception: If API call fails
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set")

    system_prompt, user_prompt = _generate_interest_prompts(topic, knowledge_cutoff_date)

    # Use utils function but tell it to use openrouter
    # We need to import from utils
    from utils import query_llm_for_numeric_value_openrouter

    return query_llm_for_numeric_value_openrouter(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=100,
        value_name="interest score"
    )


def query_divisiveness_openrouter(
    topic: str,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None
) -> float:
    """
    Query LLM via OpenRouter API to get a divisiveness score for a topic.

    Args:
        topic: The topic or subject to evaluate
        model: OpenRouter model identifier (default: "openai/gpt-4o-mini")
                Format: "provider/model-name" (e.g., "anthropic/claude-sonnet-4")
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY environment variable)
        temperature: Sampling temperature (default: 0.5)
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        float: Divisiveness score between 0 and 1

    Raises:
        ValueError: If unable to extract a valid score from the response or API key not found
        Exception: If API call fails
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set")

    system_prompt, user_prompt = _generate_divisiveness_prompts(topic, knowledge_cutoff_date)

    # Use utils function but tell it to use openrouter
    from utils import query_llm_for_numeric_value_openrouter

    return query_llm_for_numeric_value_openrouter(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=100,
        value_name="divisiveness score"
    )


def get_buzz_score(
    topic: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get both interest and divisiveness scores for a topic.

    Provider is automatically inferred from model name:
    - Models containing 'gpt', 'o1', or 'o3' use OpenAI
    - Models containing 'claude' use Anthropic

    Args:
        topic: The topic or subject to evaluate
        model: Model to use (default: "gpt-4o-mini")
        api_key: API key (if None, uses environment variable for inferred provider)
        temperature: Sampling temperature (default: 0.5)
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        dict: Contains:
            - 'topic': The topic evaluated
            - 'interest': Interest score (0-1)
            - 'divisiveness': Divisiveness score (0-1)
            - 'buzz': Combined buzz score (0-1), calculated as interest * divisiveness
            - 'provider': Provider used
            - 'model': Model used
            - 'knowledge_cutoff_date': Knowledge cutoff date used (if any)

    Raises:
        ValueError: If provider cannot be inferred or API key not found
    """
    provider = _infer_provider_from_model(model)

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

    # Query for both metrics
    interest = query_interest(topic, api_key, model, temperature, knowledge_cutoff_date)
    divisiveness = query_divisiveness(topic, api_key, model, temperature, knowledge_cutoff_date)

    # Calculate combined buzz score (interest * divisiveness)
    # High buzz requires both high interest AND high divisiveness
    buzz = interest * divisiveness

    return {
        "topic": topic,
        "interest": interest,
        "divisiveness": divisiveness,
        "buzz": buzz,
        "provider": provider,
        "model": model,
        "knowledge_cutoff_date": knowledge_cutoff_date
    }


def get_buzz_score_openrouter(
    topic: str,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get both interest and divisiveness scores for a topic using OpenRouter.

    Args:
        topic: The topic or subject to evaluate
        model: OpenRouter model identifier (default: "openai/gpt-4o-mini")
                Format: "provider/model-name" (e.g., "anthropic/claude-sonnet-4")
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY environment variable)
        temperature: Sampling temperature (default: 0.5)
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        dict: Contains:
            - 'topic': The topic evaluated
            - 'interest': Interest score (0-1)
            - 'divisiveness': Divisiveness score (0-1)
            - 'buzz': Combined buzz score (0-1), calculated as interest * divisiveness
            - 'model': Model used
            - 'knowledge_cutoff_date': Knowledge cutoff date used (if any)

    Raises:
        ValueError: If API key not found
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set")

    # Query for both metrics
    interest = query_interest_openrouter(topic, model, api_key, temperature, knowledge_cutoff_date)
    divisiveness = query_divisiveness_openrouter(topic, model, api_key, temperature, knowledge_cutoff_date)

    # Calculate combined buzz score (interest * divisiveness)
    # High buzz requires both high interest AND high divisiveness
    buzz = interest * divisiveness

    return {
        "topic": topic,
        "interest": interest,
        "divisiveness": divisiveness,
        "buzz": buzz,
        "model": model,
        "knowledge_cutoff_date": knowledge_cutoff_date
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
