# Utils Module

Shared utility functions for LLM response parsing and querying.

## Overview

The `utils.py` module provides common functionality used across multiple modules in this project, including:
- Numeric value extraction from LLM responses
- Unified LLM querying interface

## Functions

### `extract_numeric_value(text, value_name="value")`

Unified function to extract numeric values (0-1) from LLM responses in various formats.

**Supported Formats:**
- **Fractions**: `1/4`, `3/4`, `2 / 5` → 0.25, 0.75, 0.4
- **Percentages**: `25%`, `75.5%`, `100%` → 0.25, 0.755, 1.0
- **Decimals**: `0.25`, `0.75`, `0.5` → 0.25, 0.75, 0.5
- **Edge cases**: `0`, `1`, `0.0`, `1.0` → 0.0, 1.0, 0.0, 1.0

**Priority Order:**
1. Fractions (highest priority)
2. Percentages
3. Decimals (lowest priority)

**Parameters:**
- `text` (str): Response text from LLM
- `value_name` (str): Name of the value for error messages (default: "value")

**Returns:**
- `float`: Extracted value between 0 and 1

**Raises:**
- `ValueError`: If no valid value found in the expected range

**Usage:**
```python
from utils import extract_numeric_value

# Extract from different formats
prob = extract_numeric_value("The probability is 25%", "probability")  # 0.25
score = extract_numeric_value("Score: 0.75", "score")  # 0.75
ratio = extract_numeric_value("1/4 chance", "ratio")  # 0.25
```

### `query_llm_for_numeric_value(...)`

Unified function to query an LLM and extract a numeric value from the response.

Handles the common pattern of:
1. Sending a prompt to an LLM (OpenAI or Claude)
2. Getting the text response
3. Extracting a numeric value (0-1) from that response

**Parameters:**
- `system_prompt` (str): System prompt defining the LLM's role
- `user_prompt` (str): User prompt with the actual question/task
- `provider` (str): Either "openai" or "claude"
- `api_key` (str): API key for the provider
- `model` (str): Model identifier to use
- `temperature` (float): Sampling temperature (default: 0.7)
- `max_tokens` (int): Maximum tokens in response (default: 500)
- `value_name` (str): Name of value being extracted (default: "value")

**Returns:**
- `float`: Extracted numeric value between 0 and 1

**Usage:**
```python
from utils import query_llm_for_numeric_value

probability = query_llm_for_numeric_value(
    system_prompt="You are a probability estimator",
    user_prompt="What is the probability of rain?",
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini",
    value_name="probability"
)
```

### `query_llm_for_text(...)`

Unified function to query an LLM and return raw text response (not parsed for numeric values).

**Parameters:**
- Same as `query_llm_for_numeric_value` except `value_name` is not needed
- `max_tokens` defaults to 200 instead of 500

**Returns:**
- `str`: Raw text response from the LLM

**Usage:**
```python
from utils import query_llm_for_text

reworded = query_llm_for_text(
    system_prompt="You rephrase questions",
    user_prompt="Rephrase: What is the weather?",
    provider="claude",
    api_key="sk-ant-...",
    model="claude-sonnet-4-5-20250929"
)
```

## Consolidation Benefits

By consolidating the extraction logic into `utils.py`:

1. **DRY Principle**: No duplication between `probability_estimator.py` and `buzz.py`
2. **Single Source of Truth**: One place to fix bugs or add features
3. **Consistent Behavior**: Both modules extract values the same way
4. **Easier Testing**: Comprehensive tests in one place (`tests/test_utils.py`)
5. **Reusability**: Any future modules can use these utilities

## Migration Summary

### Before
- `probability_estimator.py` had `_extract_probability(text)`
- `buzz.py` had `_extract_score(text)`
- Both implemented similar regex-based extraction

### After
- Both modules import `extract_numeric_value(text, value_name)` from `utils.py`
- Single unified implementation with better error messages
- Consistent behavior across all modules

## Testing

Comprehensive test suite in `tests/test_utils.py`:

```bash
pytest tests/test_utils.py -v
```

Tests cover:
- All supported formats (fractions, percentages, decimals)
- Edge cases (0, 1, boundaries)
- Error handling (invalid text)
- Multiple values (priority order)
- Custom value names in error messages
- Out-of-range value handling

## Dependencies

- `re`: For regex pattern matching
- `openai`: For OpenAI API calls (installed via requirements.txt)
- `anthropic`: For Claude API calls (installed via requirements.txt)
