"""
Probability Estimator using OpenAI and Claude APIs

This module provides functions to query LLMs for probability estimates of events.
"""

import json
from typing import Optional, Dict, Any, List, Tuple
import os
import numpy as np
from utils import extract_numeric_value, query_llm_for_numeric_value, query_llm_for_text, generate_knowledge_cutoff_instruction
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel


# Global variable for additional instructions to inject into prompts
ADDITIONAL_INSTRUCTIONS = 'Consider any recent news or public opinion polls in your response.  Please just give your best guess numeric probability without any additional explanations.'


def _generate_probability_prompts(prompt: str, knowledge_cutoff_date: Optional[str] = None) -> Tuple[str, str]:
    """
    Generate system and user prompts for probability estimation.

    Args:
        prompt: The user's probability question
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are a probability estimation assistant. When asked about the probability of an event,
provide a numerical estimate between 0 and 1 (or 0% to 100%).

Your response MUST include a clear probability estimate. Format your response to include the probability
prominently, for example: "Probability: 0.25" or "Estimated probability: 25%" or simply "0.25".

Provide brief reasoning for your estimate, but always include a specific numerical probability."""

    # Add knowledge cutoff instruction if provided
    system_prompt += generate_knowledge_cutoff_instruction(knowledge_cutoff_date)

    user_prompt = f"{prompt}\n\n{ADDITIONAL_INSTRUCTIONS}"

    return system_prompt, user_prompt


def _generate_reword_prompts(original_prompt: str, temperature: float, knowledge_cutoff_date: Optional[str] = None) -> Tuple[str, str]:
    """
    Generate system and user prompts for rewording a question.

    Args:
        original_prompt: The original question to reword
        temperature: Controls flexibility of rewording
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = "You are a helpful assistant that rephrases questions. Only return the rephrased question, nothing else."

    # Add knowledge cutoff instruction if provided
    system_prompt += generate_knowledge_cutoff_instruction(knowledge_cutoff_date)

    # Adjust instructions based on temperature
    if temperature < 0.3:
        instruction = "Slightly rephrase the following question while preserving its exact meaning. Make only minimal changes to word choice or structure."
    elif temperature < 0.7:
        instruction = "Rephrase the following question in a different way while keeping the same core meaning. You can vary the wording and sentence structure."
    else:
        instruction = "Rephrase the following question in a significantly different way. Feel free to use different wording, structure, and style while asking about the same underlying event."

    user_prompt = f"{instruction}\n\nOriginal question: {original_prompt}"

    return system_prompt, user_prompt


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


def query_probability(
    prompt: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    knowledge_cutoff_date: Optional[str] = None
) -> float:
    """
    Query LLM API to get a probability estimate for an event.

    Provider is automatically inferred from model name:
    - Models containing 'gpt', 'o1', or 'o3' use OpenAI
    - Models containing 'claude' use Anthropic

    Args:
        prompt: User's question about probability (e.g., "What is the probability that Kamala Harris runs for office?")
        api_key: API key for the inferred provider
        model: Model to use (default: "gpt-4o-mini")
        temperature: Sampling temperature for API call (default: 0.7)
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        float: Probability between 0 and 1

    Raises:
        ValueError: If unable to extract a valid probability from the response or cannot infer provider
        Exception: If API call fails

    Examples:
        >>> query_probability("Will it rain?", api_key, model="gpt-4o-mini")
        0.45
        >>> query_probability("Will it rain?", api_key, model="claude-sonnet-4-5-20250929")
        0.42
    """
    provider = _infer_provider_from_model(model)
    system_prompt, user_prompt = _generate_probability_prompts(prompt, knowledge_cutoff_date)

    return query_llm_for_numeric_value(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=500,
        value_name="probability"
    )


async def reword_prompt(
    original_prompt: str,
    temperature: float = 0.5,
    api_key: Optional[str] = None,
    model: str = "openai/gpt-4o-mini",
    knowledge_cutoff_date: Optional[str] = None
) -> str:
    """
    Reword a prompt using OpenRouter API with variable flexibility based on temperature.

    Uses Pydantic AI with OpenRouter to access multiple model providers through a unified interface.

    Args:
        original_prompt: The original question to reword
        temperature: Controls flexibility (0 = no change, 1.0 = very loose reword)
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY environment variable)
        model: OpenRouter model identifier (default: "openai/gpt-4o-mini")
                Format: "provider/model-name" (e.g., "anthropic/claude-sonnet-4")
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        str: Reworded prompt

    Raises:
        ValueError: If API key not provided and environment variable not set
        Exception: If API call fails
    """
    # Temperature of 0 means return original prompt unchanged
    if temperature == 0:
        return original_prompt

    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set")

    # Configure OpenRouter model
    openrouter_model = OpenAIChatModel(
        model,
        provider='openrouter',
    )

    # Set API key via environment (Pydantic AI reads OPENROUTER_API_KEY by default for openrouter provider)
    original_key = os.environ.get('OPENROUTER_API_KEY')
    os.environ['OPENROUTER_API_KEY'] = api_key

    try:
        # Generate system and user prompts for rewording
        system_prompt, user_prompt = _generate_reword_prompts(original_prompt, temperature, knowledge_cutoff_date)

        # Create reword agent with system prompt
        reword_agent = Agent(openrouter_model, system_prompt=system_prompt)

        # Use Pydantic AI to reword
        reword_result = await reword_agent.run(
            user_prompt,
            message_history=[],
        )

        return str(reword_result.output).strip()
    finally:
        # Restore original environment variable
        if original_key is None:
            os.environ.pop('OPENROUTER_API_KEY', None)
        else:
            os.environ['OPENROUTER_API_KEY'] = original_key


# Compatible OpenRouter model identifiers
OPENROUTER_MODELS = [
    # OpenAI Models
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/o1",
    "openai/o1-mini",

    # Anthropic Claude Models
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-sonnet",

    # Qwen Models
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwq-32b-preview",
]


async def get_probability_distribution(
    prompt: str,
    n_samples: int = 10,
    reword_temperature: float = 0.5,
    prompt_temperature: float = 0.7,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    knowledge_cutoff_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a probability distribution by repeatedly querying an LLM via OpenRouter with reworded prompts.

    Uses Pydantic AI with OpenRouter to access multiple model providers through a unified interface.

    Args:
        prompt: Original probability question
        n_samples: Number of samples to collect (default: 10)
        reword_temperature: Reword flexibility (0 = no reword, 1.0 = very loose reword, default: 0.5)
        prompt_temperature: Sampling temperature for probability queries (default: 0.7)
        model: OpenRouter model identifier (default: "openai/gpt-4o-mini")
                Format: "provider/model-name" (e.g., "anthropic/claude-sonnet-4")
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY environment variable)
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        dict: Contains:
            - 'probabilities': List of probability estimates
            - 'reworded_prompts': List of reworded prompts used
            - 'model': Model used
            - 'n_samples': Number of samples
            - 'reword_temperature': Reword temperature used
            - 'prompt_temperature': Prompt temperature used
            - 'knowledge_cutoff_date': Knowledge cutoff date used (if any)

    Raises:
        ValueError: If API key not found
        ImportError: If pydantic-ai not installed

    Examples:
        >>> result = await get_probability_distribution(
        ...     "Will it rain tomorrow?",
        ...     model="anthropic/claude-sonnet-4",
        ...     n_samples=20
        ... )
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set")

    # Configure OpenRouter model
    # Note: The provider parameter is a Pydantic AI feature that configures the base URL automatically
    openrouter_model = OpenAIChatModel(
        model,
        provider='openrouter',
    )

    # Set API key via environment (Pydantic AI reads OPENROUTER_API_KEY by default for openrouter provider)
    original_key = os.environ.get('OPENROUTER_API_KEY')
    os.environ['OPENROUTER_API_KEY'] = api_key

    try:
        probabilities = []
        reworded_prompts = []

        # Generate system prompts (they remain constant across samples)
        prob_system_prompt, _ = _generate_probability_prompts("", knowledge_cutoff_date)

        # Create agents with system prompts
        probability_agent = Agent(openrouter_model, system_prompt=prob_system_prompt)

        for i in range(n_samples):
            # Reword the prompt (first iteration uses original if reword_temperature > 0)
            if i == 0 and reword_temperature > 0:
                # Always include the original prompt as the first sample
                reworded = prompt
            else:
                # Generate reword system and user prompts
                reword_system_prompt, reword_user_prompt = _generate_reword_prompts(prompt, reword_temperature, knowledge_cutoff_date)

                # Create reword agent with system prompt
                reword_agent = Agent(openrouter_model, system_prompt=reword_system_prompt)

                # Use Pydantic AI to reword
                reword_result = await reword_agent.run(
                    reword_user_prompt,
                    message_history=[],
                )
                reworded = str(reword_result.output).strip()

            reworded_prompts.append(reworded)

            # Generate user prompt for probability (system prompt already set on agent)
            _, user_prompt = _generate_probability_prompts(reworded, knowledge_cutoff_date)

            # Query for probability with prompt_temperature
            probability_result = await probability_agent.run(
                user_prompt,
                message_history=[],
            )

            # Extract probability from response
            response_text = str(probability_result.output)
            prob = extract_numeric_value(response_text, "probability")
            probabilities.append(prob)

        # Calculate statistics
        probs_array = np.array(probabilities)

        return {
            "probabilities": probabilities,
            "reworded_prompts": reworded_prompts,
            "model": model,
            "n_samples": n_samples,
            "reword_temperature": reword_temperature,
            "prompt_temperature": prompt_temperature,
            "knowledge_cutoff_date": knowledge_cutoff_date
        }
    finally:
        # Restore original environment variable
        if original_key is None:
            os.environ.pop('OPENROUTER_API_KEY', None)
        else:
            os.environ['OPENROUTER_API_KEY'] = original_key


async def get_probability_distribution_legacy(
    prompt: str,
    n_samples: int = 10,
    reword_temperature: float = 0.5,
    prompt_temperature: float = 0.7,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    knowledge_cutoff_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a probability distribution by repeatedly querying an LLM with reworded prompts via OpenRouter.

    LEGACY FUNCTION: This function has been superseded by get_probability_distribution.
    Uses OpenRouter API for unified access to multiple providers.

    Args:
        prompt: Original probability question
        n_samples: Number of samples to collect (default: 10)
        reword_temperature: Reword flexibility (0 = no reword, 1.0 = very loose reword, default: 0.5)
        prompt_temperature: Sampling temperature for probability queries (default: 0.7)
        model: OpenRouter model identifier (default: "openai/gpt-4o-mini")
                Format: "provider/model-name" (e.g., "anthropic/claude-sonnet-4")
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY environment variable)
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        dict: Contains:
            - 'probabilities': List of probability estimates
            - 'reworded_prompts': List of reworded prompts used
            - 'model': Model used
            - 'n_samples': Number of samples
            - 'reword_temperature': Reword temperature used
            - 'prompt_temperature': Prompt temperature used
            - 'knowledge_cutoff_date': Knowledge cutoff date used (if any)

    Raises:
        ValueError: If API key not found
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set")

    # Configure OpenRouter model
    openrouter_model = OpenAIChatModel(
        model,
        provider='openrouter',
    )

    # Set API key via environment
    original_key = os.environ.get('OPENROUTER_API_KEY')
    os.environ['OPENROUTER_API_KEY'] = api_key

    try:
        probabilities = []
        reworded_prompts = []

        # Generate system prompts
        prob_system_prompt, _ = _generate_probability_prompts("", knowledge_cutoff_date)

        # Create probability agent
        probability_agent = Agent(openrouter_model, system_prompt=prob_system_prompt)

        for i in range(n_samples):
            # Reword the prompt (first iteration uses original if reword_temperature > 0)
            if i == 0 and reword_temperature > 0:
                # Always include the original prompt as the first sample
                reworded = prompt
            else:
                reworded = await reword_prompt(prompt, reword_temperature, api_key, model, knowledge_cutoff_date)

            reworded_prompts.append(reworded)

            # Generate user prompt for probability
            _, user_prompt = _generate_probability_prompts(reworded, knowledge_cutoff_date)

            # Query for probability with prompt_temperature
            probability_result = await probability_agent.run(
                user_prompt,
                message_history=[],
            )

            # Extract probability from response
            response_text = str(probability_result.output)
            prob = extract_numeric_value(response_text, "probability")
            probabilities.append(prob)

        # Calculate statistics
        probs_array = np.array(probabilities)

        return {
            "probabilities": probabilities,
            "reworded_prompts": reworded_prompts,
            "model": model,
            "n_samples": n_samples,
            "reword_temperature": reword_temperature,
            "prompt_temperature": prompt_temperature,
            "knowledge_cutoff_date": knowledge_cutoff_date
        }
    finally:
        # Restore original environment variable
        if original_key is None:
            os.environ.pop('OPENROUTER_API_KEY', None)
        else:
            os.environ['OPENROUTER_API_KEY'] = original_key


def get_probability_estimate(
    prompt: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    knowledge_cutoff_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get a probability estimate from an LLM.

    Provider is automatically inferred from model name:
    - Models containing 'gpt', 'o1', or 'o3' use OpenAI
    - Models containing 'claude' use Anthropic

    Args:
        prompt: User's question about probability
        model: Model to use (default: "gpt-4o-mini")
        api_key: API key (if None, uses environment variable for inferred provider)
        temperature: Sampling temperature for API call (default: 0.7)
        knowledge_cutoff_date: Optional date to constrain knowledge (e.g., "January 2024")

    Returns:
        dict: Contains 'probability' (float), 'provider' (str), 'model' (str), and 'knowledge_cutoff_date' (str or None)

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

    probability = query_probability(prompt, api_key, model, temperature, knowledge_cutoff_date)

    return {
        "probability": probability,
        "provider": provider,
        "model": model,
        "knowledge_cutoff_date": knowledge_cutoff_date
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
