"""
Probability Estimator using OpenAI and Claude APIs

This module provides functions to query LLMs for probability estimates of events.
"""

import json
from typing import Optional, Dict, Any, List, Tuple
import os
import numpy as np
from utils import extract_numeric_value, query_llm_for_numeric_value, query_llm_for_text


# Global variable for additional instructions to inject into prompts
ADDITIONAL_INSTRUCTIONS = 'Consider any recent news or public opinion polls in your response.  Please just give your best guess numeric probability without any additional explanations.'


def _generate_probability_prompts(prompt: str) -> Tuple[str, str]:
    """
    Generate system and user prompts for probability estimation.

    Args:
        prompt: The user's probability question

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are a probability estimation assistant. When asked about the probability of an event,
provide a numerical estimate between 0 and 1 (or 0% to 100%).

Your response MUST include a clear probability estimate. Format your response to include the probability
prominently, for example: "Probability: 0.25" or "Estimated probability: 25%" or simply "0.25".

Provide brief reasoning for your estimate, but always include a specific numerical probability."""

    user_prompt = f"{prompt}\n\n{ADDITIONAL_INSTRUCTIONS}"

    return system_prompt, user_prompt


def _generate_reword_prompts(original_prompt: str, temperature: float) -> Tuple[str, str]:
    """
    Generate system and user prompts for rewording a question.

    Args:
        original_prompt: The original question to reword
        temperature: Controls flexibility of rewording

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = "You are a helpful assistant that rephrases questions. Only return the rephrased question, nothing else."

    # Adjust instructions based on temperature
    if temperature < 0.3:
        instruction = "Slightly rephrase the following question while preserving its exact meaning. Make only minimal changes to word choice or structure."
    elif temperature < 0.7:
        instruction = "Rephrase the following question in a different way while keeping the same core meaning. You can vary the wording and sentence structure."
    else:
        instruction = "Rephrase the following question in a significantly different way. Feel free to use different wording, structure, and style while asking about the same underlying event."

    user_prompt = f"{instruction}\n\nOriginal question: {original_prompt}"

    return system_prompt, user_prompt


def query_openai_probability(prompt: str, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> float:
    """
    Query OpenAI API to get a probability estimate for an event.

    Args:
        prompt: User's question about probability (e.g., "What is the probability that Kamala Harris runs for office?")
        api_key: OpenAI API key
        model: Model to use (default: "gpt-4o-mini")
        temperature: Sampling temperature for API call (default: 0.7)

    Returns:
        float: Probability between 0 and 1

    Raises:
        ValueError: If unable to extract a valid probability from the response
        Exception: If API call fails
    """
    system_prompt, user_prompt = _generate_probability_prompts(prompt)

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider="openai",
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=500,
        value_name="probability"
    )


def query_claude_probability(prompt: str, api_key: str, model: str = "claude-sonnet-4-5-20250929", temperature: float = 0.7) -> float:
    """
    Query Claude API to get a probability estimate for an event.

    Args:
        prompt: User's question about probability (e.g., "What is the probability that Kamala Harris runs for office?")
        api_key: Anthropic API key
        model: Model to use (default: "claude-sonnet-4-5-20250929")
        temperature: Sampling temperature for API call (default: 0.7)

    Returns:
        float: Probability between 0 and 1

    Raises:
        ValueError: If unable to extract a valid probability from the response
        Exception: If API call fails
    """
    system_prompt, user_prompt = _generate_probability_prompts(prompt)

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider="claude",
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=500,
        value_name="probability"
    )


def reword_prompt_openai(
    original_prompt: str,
    temperature: float = 0.5,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Reword a prompt using OpenAI API with variable flexibility based on temperature.

    Args:
        original_prompt: The original question to reword
        temperature: Controls flexibility (0 = no change, 1.0 = very loose reword)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY environment variable)
        model: Model to use (default: "gpt-4o-mini")

    Returns:
        str: Reworded prompt

    Raises:
        Exception: If API call fails
    """
    # Temperature of 0 means return original prompt unchanged
    if temperature == 0:
        return original_prompt

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")

    system_prompt, user_prompt = _generate_reword_prompts(original_prompt, temperature)

    result = query_llm_for_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider="openai",
        api_key=api_key,
        model=model,
        temperature=min(temperature * 1.5, 1.0),  # Scale up temperature for API
        max_tokens=200
    )

    return result.strip()


def reword_prompt_claude(
    original_prompt: str,
    temperature: float = 0.5,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-5-20250929"
) -> str:
    """
    Reword a prompt using Claude API with variable flexibility based on temperature.

    Args:
        original_prompt: The original question to reword
        temperature: Controls flexibility (0 = no change, 1.0 = very loose reword)
        api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY environment variable)
        model: Model to use (default: "claude-sonnet-4-5-20250929")

    Returns:
        str: Reworded prompt

    Raises:
        Exception: If API call fails
    """
    # Temperature of 0 means return original prompt unchanged
    if temperature == 0:
        return original_prompt

    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")

    system_prompt, user_prompt = _generate_reword_prompts(original_prompt, temperature)

    result = query_llm_for_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider="claude",
        api_key=api_key,
        model=model,
        temperature=min(temperature * 1.5, 1.0),  # Scale up temperature for API
        max_tokens=200
    )

    return result.strip()


def get_probability_distribution(
    prompt: str,
    n_samples: int = 10,
    reword_temperature: float = 0.5,
    prompt_temperature: float = 0.7,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a probability distribution by repeatedly querying an LLM with reworded prompts.

    Args:
        prompt: Original probability question
        n_samples: Number of samples to collect (default: 10)
        reword_temperature: Reword flexibility (0 = no reword, 1.0 = very loose reword, default: 0.5)
        prompt_temperature: Sampling temperature for probability queries (default: 0.7)
        provider: Either "openai" or "claude" (default: "openai")
        api_key: API key (if None, will try to read from environment variables)
        model: Model to use (if None, uses default for provider)

    Returns:
        dict: Contains:
            - 'probabilities': List of probability estimates
            - 'reworded_prompts': List of reworded prompts used
            - 'provider': Provider used
            - 'model': Model used
            - 'n_samples': Number of samples
            - 'reword_temperature': Reword temperature used
            - 'prompt_temperature': Prompt temperature used

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

    # Select reword function based on provider
    reword_func = reword_prompt_openai if provider == "openai" else reword_prompt_claude

    # Select probability query function based on provider
    query_func = query_openai_probability if provider == "openai" else query_claude_probability

    probabilities = []
    reworded_prompts = []

    for i in range(n_samples):
        # Reword the prompt (first iteration uses original if reword_temperature > 0)
        if i == 0 and reword_temperature > 0:
            # Always include the original prompt as the first sample
            reworded = prompt
        else:
            reworded = reword_func(prompt, reword_temperature, api_key, model)

        reworded_prompts.append(reworded)

        # Query for probability with prompt_temperature
        prob = query_func(reworded, api_key, model, prompt_temperature)
        probabilities.append(prob)

    # Calculate statistics
    probs_array = np.array(probabilities)

    return {
        "probabilities": probabilities,
        "reworded_prompts": reworded_prompts,
        "provider": provider,
        "model": model,
        "n_samples": n_samples,
        "reword_temperature": reword_temperature,
        "prompt_temperature": prompt_temperature
    }


def get_probability_estimate(
    prompt: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Unified interface to get probability estimates from either OpenAI or Claude.

    Args:
        prompt: User's question about probability
        provider: Either "openai" or "claude" (default: "openai")
        api_key: API key (if None, will try to read from environment variables)
        model: Model to use (if None, uses default for provider)
        temperature: Sampling temperature for API call (default: 0.7)

    Returns:
        dict: Contains 'probability' (float), 'provider' (str), and 'model' (str)

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

    # Query the appropriate API
    if provider == "openai":
        model = model or "gpt-4o-mini"
        probability = query_openai_probability(prompt, api_key, model, temperature)
    else:  # claude
        model = model or "claude-sonnet-4-5-20250929"
        probability = query_claude_probability(prompt, api_key, model, temperature)

    return {
        "probability": probability,
        "provider": provider,
        "model": model
    }


if __name__ == "__main__":
    # Example usage
    import sys

    example_prompt = "What is the probability that Kamala Harris runs for president again?"

    print("Probability Estimator - Example Usage\n")
    print(f"Example prompt: {example_prompt}\n")

    # You can test by providing API keys as command line arguments
    # python probability_estimator.py <provider> <api_key>

    if len(sys.argv) >= 3:
        provider = sys.argv[1]
        api_key = sys.argv[2]

        try:
            result = get_probability_estimate(example_prompt, provider=provider, api_key=api_key)
            print(f"Provider: {result['provider']}")
            print(f"Model: {result['model']}")
            print(f"Probability: {result['probability']:.2%} ({result['probability']:.3f})")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("To test, run:")
        print("  python probability_estimator.py openai <your-openai-api-key>")
        print("  python probability_estimator.py claude <your-anthropic-api-key>")
        print("\nOr set environment variables OPENAI_API_KEY or ANTHROPIC_API_KEY")
