"""
Utility functions for LLM response parsing and querying.

This module provides helper functions for extracting numeric values from LLM responses
and a unified interface for querying different LLM providers.
"""

import re
from typing import Optional
import os


def generate_knowledge_cutoff_instruction(knowledge_cutoff_date: Optional[str] = None) -> str:
    """
    Generate instruction text for knowledge cutoff date constraint.

    This function creates a standardized instruction that can be appended to system prompts
    to enforce a knowledge cutoff date, ensuring the LLM does not use information after
    the specified date.

    Args:
        knowledge_cutoff_date: Date string (e.g., "January 2024", "2024-01-01", "Jan 1, 2024").
                              If None, returns empty string (no constraint).

    Returns:
        str: Instruction text to append to system prompt, or empty string if no cutoff specified.

    Examples:
        >>> generate_knowledge_cutoff_instruction("January 2024")
        '\\n\\nIMPORTANT: You must not use any information, events, or data from after January 2024...'

        >>> generate_knowledge_cutoff_instruction(None)
        ''
    """
    if knowledge_cutoff_date is None:
        return ""

    return f"\n\nIMPORTANT: You must not use any information, events, or data from after {knowledge_cutoff_date}. Your knowledge is limited to information available up to and including {knowledge_cutoff_date}."


def extract_numeric_value(text: str, value_name: str = "value") -> float:
    """
    Extract a numeric value (0-1) from text response.

    This unified function can extract probabilities, scores, or any other numeric
    value between 0 and 1 from LLM responses in various formats.

    Looks for patterns like:
    - "0.25" or "0.75" (decimal)
    - "25%" or "75%" (percentage)
    - "1/4" or "3/4" (fraction)
    - "Score: 0.25" or "Probability: 0.75" (labeled)

    Args:
        text: Response text from LLM
        value_name: Name of the value being extracted (for error messages)

    Returns:
        float: Extracted value between 0 and 1

    Raises:
        ValueError: If no valid value found in the expected range

    Examples:
        >>> extract_numeric_value("The probability is 25%")
        0.25
        >>> extract_numeric_value("Score: 0.75")
        0.75
        >>> extract_numeric_value("1/4 chance")
        0.25
    """
    # Try to find fractions first (e.g., "3/4", "1/2") - highest priority
    # Fractions are checked first to avoid confusion with decimals
    fraction_pattern = r'(\d+)\s*/\s*(\d+)'
    fraction_matches = re.findall(fraction_pattern, text)

    if fraction_matches:
        for numerator, denominator in fraction_matches:
            value = float(numerator) / float(denominator)
            if 0 <= value <= 1:
                return value

    # Try to find percentage (e.g., "25%", "75.5%")
    percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
    percent_matches = re.findall(percent_pattern, text)

    if percent_matches:
        for match in percent_matches:
            value = float(match) / 100.0
            if 0 <= value <= 1:
                return value

    # Try to find decimal value (e.g., "0.25", "0.75", "0", "1")
    # Match standalone numbers between 0 and 1
    decimal_pattern = r'\b(0\.\d+|1\.0+|^0$|^1$)\b'
    decimal_matches = re.findall(decimal_pattern, text)

    if decimal_matches:
        for match in decimal_matches:
            value = float(match)
            if 0 <= value <= 1:
                return value

    # If nothing found, raise an error
    raise ValueError(
        f"Could not extract a valid {value_name} (0-1) from response: {text[:200]}..."
    )


def query_llm_for_numeric_value(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
    value_name: str = "value"
) -> float:
    """
    Unified function to query an LLM and extract a numeric value from the response.

    This function handles the common pattern of:
    1. Sending a prompt to an LLM (OpenAI or Claude)
    2. Getting the text response
    3. Extracting a numeric value (0-1) from that response

    Args:
        system_prompt: The system prompt that defines the LLM's role
        user_prompt: The user prompt with the actual question/task
        provider: Either "openai" or "claude"
        api_key: API key for the provider
        model: Model identifier to use
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: 500)
        value_name: Name of value being extracted, for error messages (default: "value")

    Returns:
        float: Extracted numeric value between 0 and 1

    Raises:
        ValueError: If provider is invalid or value cannot be extracted
        Exception: If API call fails
        ImportError: If required package is not installed

    Examples:
        >>> query_llm_for_numeric_value(
        ...     "You are a probability estimator",
        ...     "What is the probability of rain?",
        ...     "openai",
        ...     "sk-...",
        ...     "gpt-4o-mini"
        ... )
        0.45
    """
    provider = provider.lower()

    if provider not in ["openai", "claude"]:
        raise ValueError("Provider must be either 'openai' or 'claude'")

    # Query the appropriate provider
    if provider == "openai":
        response_text = _query_openai(
            system_prompt, user_prompt, api_key, model, temperature, max_tokens
        )
    else:  # claude
        response_text = _query_claude(
            system_prompt, user_prompt, api_key, model, temperature, max_tokens
        )

    # Extract and return the numeric value
    return extract_numeric_value(response_text, value_name)


def _query_openai(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    Internal function to query OpenAI API.

    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        api_key: OpenAI API key
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        str: Response text from the API

    Raises:
        ImportError: If openai package not installed
        Exception: If API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"OpenAI API call failed: {str(e)}")


def _query_claude(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    Internal function to query Claude API.

    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        api_key: Anthropic API key
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        str: Response text from the API

    Raises:
        ImportError: If anthropic package not installed
        Exception: If API call fails
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

    client = Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.content[0].text

    except Exception as e:
        raise Exception(f"Claude API call failed: {str(e)}")


def _query_openrouter(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    Internal function to query OpenRouter API.

    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        api_key: OpenRouter API key
        model: Model identifier (format: "provider/model-name")
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        str: Response text from the API

    Raises:
        ImportError: If openai package not installed
        Exception: If API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")

    # OpenRouter uses OpenAI-compatible API
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"OpenRouter API call failed: {str(e)}")


def query_llm_for_numeric_value_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
    value_name: str = "value"
) -> float:
    """
    Query an LLM via OpenRouter and extract a numeric value from the response.

    Args:
        system_prompt: The system prompt that defines the LLM's role
        user_prompt: The user prompt with the actual question/task
        model: OpenRouter model identifier (format: "provider/model-name")
        api_key: OpenRouter API key
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: 500)
        value_name: Name of value being extracted, for error messages (default: "value")

    Returns:
        float: Extracted numeric value between 0 and 1

    Raises:
        ValueError: If value cannot be extracted
        Exception: If API call fails
        ImportError: If required package is not installed

    Examples:
        >>> query_llm_for_numeric_value_openrouter(
        ...     "You are a probability estimator",
        ...     "What is the probability of rain?",
        ...     "openai/gpt-4o-mini",
        ...     "sk-or-..."
        ... )
        0.45
    """
    response_text = _query_openrouter(
        system_prompt, user_prompt, api_key, model, temperature, max_tokens
    )

    # Extract and return the numeric value
    return extract_numeric_value(response_text, value_name)


def query_llm_for_text(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 200
) -> str:
    """
    Unified function to query an LLM and return raw text response.

    This is for cases where you want the full text response, not just a numeric value.

    Args:
        system_prompt: The system prompt that defines the LLM's role
        user_prompt: The user prompt with the actual question/task
        provider: Either "openai" or "claude"
        api_key: API key for the provider
        model: Model identifier to use
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: 200)

    Returns:
        str: Raw text response from the LLM

    Raises:
        ValueError: If provider is invalid
        Exception: If API call fails
        ImportError: If required package is not installed
    """
    provider = provider.lower()

    if provider not in ["openai", "claude"]:
        raise ValueError("Provider must be either 'openai' or 'claude'")

    # Query the appropriate provider
    if provider == "openai":
        return _query_openai(
            system_prompt, user_prompt, api_key, model, temperature, max_tokens
        )
    else:  # claude
        return _query_claude(
            system_prompt, user_prompt, api_key, model, temperature, max_tokens
        )


def query_llm_for_text_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 200
) -> str:
    """
    Query an LLM via OpenRouter and return raw text response.

    Args:
        system_prompt: The system prompt that defines the LLM's role
        user_prompt: The user prompt with the actual question/task
        model: OpenRouter model identifier (format: "provider/model-name")
        api_key: OpenRouter API key
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: 200)

    Returns:
        str: Raw text response from the LLM

    Raises:
        Exception: If API call fails
        ImportError: If required package is not installed
    """
    return _query_openrouter(
        system_prompt, user_prompt, api_key, model, temperature, max_tokens
    )
