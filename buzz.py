"""
Buzz Analyzer using OpenAI and Claude APIs

This module provides functions to quantify the "buzz" around topics by measuring:
1. Interest level (raw public attention via internet traffic, news coverage)
2. Divisiveness (how polarizing/controversial the topic is)
3. Estimated market metrics (volume, liquidity, etc.) using historical data
"""

# Standard library
import json
import os
import re
import sys
from typing import Optional, Dict, Any, List, Tuple

# Third-party
import numpy as np
import pandas as pd

# Local imports
from utils import (
    extract_numeric_value,
    query_llm_for_numeric_value,
    query_llm_for_numeric_value_openrouter,
    query_llm_for_text_openrouter,
    generate_knowledge_cutoff_instruction,
)
from polymarket_data import get_all_closed_markets


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

    raise ValueError(
        f"Cannot infer provider from model name '{model}'. Model must contain 'gpt', 'o1', 'o3', or 'claude'."
    )


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


def _generate_interest_prompts(
    topic: str, knowledge_cutoff_date: Optional[str] = None
) -> Tuple[str, str]:
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


def _generate_divisiveness_prompts(
    topic: str, knowledge_cutoff_date: Optional[str] = None
) -> Tuple[str, str]:
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
    knowledge_cutoff_date: Optional[str] = None,
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
    system_prompt, user_prompt = _generate_interest_prompts(
        topic, knowledge_cutoff_date
    )

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=100,
        value_name="interest score",
    )


def query_divisiveness(
    topic: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None,
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
    system_prompt, user_prompt = _generate_divisiveness_prompts(
        topic, knowledge_cutoff_date
    )

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=100,
        value_name="divisiveness score",
    )


def query_interest_openrouter(
    topic: str,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None,
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
            raise ValueError(
                "OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set"
            )

    system_prompt, user_prompt = _generate_interest_prompts(
        topic, knowledge_cutoff_date
    )

    return query_llm_for_numeric_value_openrouter(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=100,
        value_name="interest score",
    )


def query_divisiveness_openrouter(
    topic: str,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None,
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
            raise ValueError(
                "OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set"
            )

    system_prompt, user_prompt = _generate_divisiveness_prompts(
        topic, knowledge_cutoff_date
    )

    return query_llm_for_numeric_value_openrouter(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=100,
        value_name="divisiveness score",
    )


def get_buzz_score(
    topic: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None,
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
                raise ValueError(
                    "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
                )
        else:  # claude
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set"
                )

    # Query for both metrics
    interest = query_interest(topic, api_key, model, temperature, knowledge_cutoff_date)
    divisiveness = query_divisiveness(
        topic, api_key, model, temperature, knowledge_cutoff_date
    )

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
        "knowledge_cutoff_date": knowledge_cutoff_date,
    }


def get_buzz_score_openrouter(
    topic: str,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    knowledge_cutoff_date: Optional[str] = None,
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
            raise ValueError(
                "OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set"
            )

    # Query for both metrics
    interest = query_interest_openrouter(
        topic, model, api_key, temperature, knowledge_cutoff_date
    )
    divisiveness = query_divisiveness_openrouter(
        topic, model, api_key, temperature, knowledge_cutoff_date
    )

    # Calculate combined buzz score (interest * divisiveness)
    # High buzz requires both high interest AND high divisiveness
    buzz = interest * divisiveness

    return {
        "topic": topic,
        "interest": interest,
        "divisiveness": divisiveness,
        "buzz": buzz,
        "model": model,
        "knowledge_cutoff_date": knowledge_cutoff_date,
    }


def _select_representative_sample(
    df: pd.DataFrame,
    metric_column: str,
    n_samples: int,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Select N representative samples evenly distributed across a metric range.

    This is a helper function that can be reused for any metric-based sampling.

    Args:
        df: DataFrame with market data
        metric_column: Column name containing the metric to distribute across
        n_samples: Number of samples to select
        min_value: Optional minimum value to filter (default: use data min)
        max_value: Optional maximum value to filter (default: use data max)

    Returns:
        DataFrame with n_samples rows evenly distributed across the metric range
    """
    # Filter to rows with valid metric values
    df_valid = df[df[metric_column].notna() & (df[metric_column] > 0)].copy()

    if len(df_valid) == 0:
        raise ValueError(f"No valid data found in column '{metric_column}'")

    # Apply min/max filters if provided
    if min_value is not None:
        df_valid = df_valid[df_valid[metric_column] >= min_value]
    if max_value is not None:
        df_valid = df_valid[df_valid[metric_column] <= max_value]

    if len(df_valid) < n_samples:
        print(
            f"Warning: Only {len(df_valid)} valid samples available, requested {n_samples}"
        )
        return df_valid

    # Sort by metric
    df_valid = df_valid.sort_values(metric_column)

    # Create evenly spaced quantiles
    quantiles = np.linspace(0, 1, n_samples)

    # Select one sample from each quantile
    samples = []
    for q in quantiles:
        # Get the index closest to this quantile
        idx = int(q * (len(df_valid) - 1))
        samples.append(df_valid.iloc[idx])

    return pd.DataFrame(samples)


def _compute_daily_volume(row: pd.Series) -> Optional[float]:
    """
    Compute average daily volume for a market.

    Args:
        row: DataFrame row with 'volumeNum', 'createdAt', and 'endDate' or 'closedTime'

    Returns:
        Average daily volume, or None if cannot be computed
    """
    volume = row.get("volumeNum") or row.get("volume_num")
    if pd.isna(volume) or volume <= 0:
        return None

    # Try to get start and end dates
    start_date = row.get("created_at_parsed") or row.get("createdAt")
    end_date = (
        row.get("closed_time_parsed")
        or row.get("end_date_parsed")
        or row.get("closedTime")
        or row.get("endDate")
    )

    if pd.isna(start_date) or pd.isna(end_date):
        return None

    # Convert to datetime if needed
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date, errors="coerce")
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date, errors="coerce")

    if pd.isna(start_date) or pd.isna(end_date):
        return None

    # Compute days active
    days_active = (end_date - start_date).days

    if days_active <= 0:
        return None

    return volume / days_active


def _prepare_market_examples(
    df: pd.DataFrame, metric_name: str, compute_metric_fn: callable, n_examples: int
) -> List[Dict[str, Any]]:
    """
    Prepare examples for LLM prompt by computing metric and selecting representative samples.

    This is a generic helper that can be reused for any market metric.

    Args:
        df: DataFrame with closed market data
        metric_name: Name of the metric (e.g., "daily_volume", "liquidity")
        compute_metric_fn: Function that takes a row and returns the metric value
        n_examples: Number of examples to include

    Returns:
        List of dicts with 'question' and metric_name keys
    """
    # Compute metric for all markets
    df_with_metric = df.copy()
    df_with_metric[metric_name] = df_with_metric.apply(compute_metric_fn, axis=1)

    # Filter to valid metrics
    df_valid = df_with_metric[
        df_with_metric[metric_name].notna() & (df_with_metric[metric_name] > 0)
    ]

    if len(df_valid) == 0:
        raise ValueError(f"No markets with valid {metric_name} found")

    # Select representative sample
    sample_df = _select_representative_sample(df_valid, metric_name, n_examples)

    # Convert to list of examples
    examples = []
    for _, row in sample_df.iterrows():
        examples.append({"question": row["question"], metric_name: row[metric_name]})

    return examples


def estimate_daily_volume(
    query: str,
    n_examples: int = 20,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    print_prompt: bool = False,
    cache_dir: str = "data/polymarket",
) -> Dict[str, Any]:
    """
    Estimate the likely average daily volume for a prediction market query.

    Uses historical closed markets as examples to help the LLM make informed estimates.

    Args:
        query: The prediction market question to estimate volume for
        n_examples: Number of historical examples to include (default: 20)
        model: OpenRouter model identifier (default: "openai/gpt-4o-mini")
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY environment variable)
        temperature: Sampling temperature (default: 0.7)
        print_prompt: If True, print the full prompt before making the API call
        cache_dir: Directory containing cached closed markets data

    Returns:
        dict: Contains:
            - 'query': The input query
            - 'estimated_daily_volume': Estimated average daily volume in USD
            - 'n_examples': Number of examples used
            - 'model': Model used
            - 'examples': List of example markets used (if you want to inspect)

    Example:
        >>> result = estimate_daily_volume(
        ...     "Will Bitcoin exceed $100k by end of 2025?",
        ...     n_examples=20
        ... )
        >>> print(f"Estimated daily volume: ${result['estimated_daily_volume']:.2f}")
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set"
            )

    # Load closed markets data
    print(f"Loading closed markets data...")
    df = get_all_closed_markets(cache_dir=cache_dir, overwrite=False)
    print(f"Loaded {len(df)} closed markets")

    # Prepare examples using the generic helper
    print(f"Selecting {n_examples} representative examples...")
    examples = _prepare_market_examples(
        df=df,
        metric_name="daily_volume",
        compute_metric_fn=_compute_daily_volume,
        n_examples=n_examples,
    )

    # Build the prompt
    system_prompt = """You are an expert at estimating prediction market metrics. You have deep knowledge of what types of topics attract trading volume on platforms like Polymarket.

Your task is to estimate the likely average daily trading volume (in USD) for a new prediction market question based on historical examples."""

    # Build examples section
    examples_text = "Here are examples of historical prediction markets and their average daily volumes:\n\n"
    for i, ex in enumerate(examples, 1):
        examples_text += f"{i}. Question: {ex['question']}\n"
        examples_text += f"   Average Daily Volume: ${ex['daily_volume']:.2f}\n\n"

    user_prompt = f"""{examples_text}

Based on these examples, estimate the likely average daily trading volume for the following new market:

Question: {query}

Consider factors like:
- Topic relevance and public interest
- Clarity and resolvability of the question
- Time horizon (shorter-term questions often get more trading)
- Comparison to similar questions in the examples above

Provide your estimate as a single number (the estimated daily volume in USD). Be specific and realistic based on the examples.

Your response should be ONLY the numeric estimate, like: 1250.50"""

    if print_prompt:
        print("\n" + "=" * 80)
        print("FULL PROMPT")
        print("=" * 80)
        print("\nSYSTEM PROMPT:")
        print(system_prompt)
        print("\nUSER PROMPT:")
        print(user_prompt)
        print("=" * 80 + "\n")

    # Query the LLM
    print("Querying LLM for volume estimate...")
    response = query_llm_for_text_openrouter(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=100,
    )

    # Parse the response directly as a number (not as 0-1 like extract_numeric_value does)
    try:
        # Try to find a number in the response
        numbers = re.findall(r"\d+\.?\d*", response)
        if numbers:
            estimated_volume = float(numbers[0])
        else:
            raise ValueError(f"No numeric value found in response: {response}")
    except Exception as e:
        raise ValueError(
            f"Could not parse volume from response: {response}. Error: {e}"
        )

    return {
        "query": query,
        "estimated_daily_volume": estimated_volume,
        "n_examples": len(examples),
        "model": model,
        "examples": examples,  # Include for inspection
    }


if __name__ == "__main__":
    # Example usage
    example_topics = [
        "Artificial Intelligence",
        "Climate Change",
        "2024 US Presidential Election",
        "The Beatles",
        "Quantum Computing",
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
            print(
                f"  Interest:      {result['interest']:.3f} ({result['interest']*100:.1f}%)"
            )
            print(
                f"  Divisiveness:  {result['divisiveness']:.3f} ({result['divisiveness']*100:.1f}%)"
            )
            print(f"  Buzz Score:    {result['buzz']:.3f} ({result['buzz']*100:.1f}%)")

            # Interpretation
            print(f"\nInterpretation:")
            if result["buzz"] > 0.5:
                print(
                    "  🔥 HIGH BUZZ - This topic has significant attention and controversy"
                )
            elif result["buzz"] > 0.25:
                print(
                    "  📊 MODERATE BUZZ - This topic has notable presence in discourse"
                )
            else:
                print("  📉 LOW BUZZ - This topic has limited attention or controversy")

        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("To test, run:")
        print('  python buzz.py openai <your-openai-api-key> "Topic Name"')
        print('  python buzz.py claude <your-anthropic-api-key> "Topic Name"')
        print("\nOr set environment variables OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print(f"\nExample topics to try:")
        for topic in example_topics:
            print(f"  - {topic}")
