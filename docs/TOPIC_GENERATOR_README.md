# Topic Generator

Generate trending topics from news and social media, along with forecasting questions that have objective future answers.

## Overview

The `topic_generator.py` module uses OpenRouter API with web search enabled (GPT-4o-mini:online as default) to:
1. Search the web for the most discussed and impactful stories from recent news (last 7 days)
2. Generate yes/no forecasting questions with specific future resolution dates
3. Return a dictionary mapping topic keywords to lists of questions

**Key Feature**: Uses OpenRouter's `:online` feature to access real-time web data and current news, ensuring questions are based on truly recent events with future resolution dates.

## Features

- **Configurable Parameters**:
  - `n_topics`: Number of trending topics to generate
  - `k_questions`: Maximum questions per topic
  - `num_days`: Number of days to look back for recent news (default: 7)
- **Smart Filtering**: Automatically filters out topics with no valid questions
- **Topic Diversity Enforcement**: Ensures topics are from different domains/subject areas (e.g., won't create separate topics for "EU AI Act" and "AI Governance" - combines them into one)
- **Question Validation**: Ensures questions are:
  - Yes/no format
  - Have future objective answers (with specific resolution dates)
  - Never use dates in the past
  - Specific and verifiable
  - Relevant to the topic
  - Prefer near-term (1-3 years) over far-future questions
- **Intelligent Context Validation** (NEW):
  - Automatically detects and removes nonsensical questions (e.g., "Will the Biden administration achieve X by 2030?" when Biden is no longer president)
  - Filters out questions with obvious answers based on current events
  - Ensures questions remain meaningful in today's context
  - Can be disabled by setting `validate_questions_flag=False`
- **Web Search Enabled**: Uses OpenRouter's `:online` feature for real-time news access
- **Date Intelligence**:
  - Automatically includes today's date in prompts
  - Uses dates from articles when mentioned
  - Generates reasonable near-term dates when not specified

## Installation

Ensure you have the required dependencies:

```bash
pip install openai  # For OpenRouter API access
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

### As a Command-Line Tool

```bash
# Generate 5 topics with up to 3 questions each (defaults)
python topic_generator.py

# Generate 2 topics with 1 question each
python topic_generator.py 2 1

# Generate 10 topics with up to 5 questions each
python topic_generator.py 10 5
```

### As a Python Module

```python
from topic_generator import generate_topics_and_questions, print_topics_and_questions

# Generate topics and questions
result = generate_topics_and_questions(n_topics=3, k_questions=2)

# Print results in a nice format
print_topics_and_questions(result)

# Or access the data directly
for topic, questions in result.items():
    print(f"Topic: {topic}")
    for question in questions:
        print(f"  - {question}")
```

### Example Output

```
================================================================================
GENERATED TOPICS AND FORECASTING QUESTIONS
================================================================================

1. AI Regulation
--------------------------------------------------------------------------------
   1) Will the EU AI Act be fully implemented by December 2025?
   2) Will the US pass comprehensive federal AI regulation by 2026?

2. Climate Summit COP29
--------------------------------------------------------------------------------
   1) Will global CO2 emissions decrease in 2025 compared to 2024?
   2) Will at least 100 countries commit to net-zero by 2040 at COP29?

3. Space Exploration
--------------------------------------------------------------------------------
   1) Will SpaceX successfully land humans on Mars by 2030?

Summary: 3 topics, 5 questions total
```

## API Reference

### `generate_topics_and_questions(n_topics, k_questions, model, api_key, temperature, enable_web_search, num_days, validate_questions_flag)`

Generate trending topics and forecasting questions with automatic validation.

**Parameters:**
- `n_topics` (int): Number of trending topics to generate (default: 5)
- `k_questions` (int): Maximum questions per topic (default: 3)
- `model` (str): OpenRouter model identifier (default: "openai/gpt-4o-mini:online")
- `api_key` (str, optional): OpenRouter API key (uses OPENROUTER_API_KEY env var if not provided)
- `temperature` (float): Sampling temperature (default: 0.7)
- `enable_web_search` (bool): Automatically add `:online` suffix for web search (default: True)
- `num_days` (int): Number of days to look back for recent news (default: 7)
- `validate_questions_flag` (bool): Enable context validation to filter out nonsensical questions (default: True)

**Returns:**
- `dict`: Mapping of topic keywords to list of forecasting questions

**Raises:**
- `ValueError`: If parameters are invalid or API key is missing
- `Exception`: If API call fails

**Example:**
```python
# Basic usage (with validation enabled by default)
result = generate_topics_and_questions(
    n_topics=2,
    k_questions=1,
    model="anthropic/claude-sonnet-4",
    temperature=0.8
)

# Disable validation if you want to keep all generated questions
result = generate_topics_and_questions(
    n_topics=5,
    k_questions=3,
    num_days=60,  # Look back 60 days
    validate_questions_flag=False  # Disable validation
)
```

### `validate_questions(topics, model, api_key, temperature)`

Validate and filter out nonsensical or obviously-answered questions.

**Parameters:**
- `topics` (dict): Dictionary mapping topics to questions
- `model` (str): OpenRouter model identifier (default: "openai/gpt-4o-mini")
- `api_key` (str, optional): OpenRouter API key
- `temperature` (float): Sampling temperature (default: 0.3 for consistency)

**Returns:**
- `dict`: Filtered topics with only valid questions

**Example:**
```python
from topic_generator import validate_questions

# Manually validate questions
topics = {
    "US Politics": [
        "Will the Biden administration achieve X by 2030?",  # May be invalid
        "Will Congress pass legislation by 2026?"
    ]
}

validated = validate_questions(topics)
# Result: Only the second question remains if Biden is no longer president
```

### `print_topics_and_questions(topics)`

Pretty print topics and questions.

**Parameters:**
- `topics` (dict): Dictionary mapping topics to questions (output from `generate_topics_and_questions`)

## Testing

Run the unit tests:

```bash
# Run all tests
pytest tests/test_topic_generator.py -v

# Run only the basic N=2, k=1 test (as specified in requirements)
pytest tests/test_topic_generator.py::TestGenerateTopicsAndQuestions::test_basic_functionality_n2_k1 -v

# Run including integration test (requires OPENROUTER_API_KEY)
pytest tests/test_topic_generator.py -v --runintegration
```

### Test Coverage

The test suite includes:
- **Basic functionality test with N=2, k=1** (as requested in requirements)
- Parameter validation tests
- JSON parsing tests (including malformed responses)
- Filtering tests (non-string questions, empty topics)
- Custom model and temperature tests
- Integration test with real API call (skipped if no API key)

## Topic and Question Criteria

### Topic Diversity Requirements

Topics must be **substantially different** from each other:

**❌ DON'T DO THIS** (Duplicate topics):
- Topic 1: "EU AI Act Implementation"
- Topic 2: "AI Governance and Regulation"

  *Problem*: Both are about EU AI regulation. Should be **ONE topic** with multiple questions.

**✅ DO THIS INSTEAD** (Diverse topics):
- Topic 1: "EU AI Act Implementation"
  - Questions: "Will the EU fully implement the AI Act by August 2026?" AND "Will prohibitions be enforced by February 2025?"
- Topic 2: "US-China Trade Relations" (completely different domain)
  - Questions: "Will US impose new tariffs by June 2026?"

### Question Criteria

Generated questions must satisfy ALL of the following:

1. **Yes/No Format**: Must be answerable with yes or no
2. **Future Answer**: The answer is not currently known or agreed upon
3. **Objective**: Will have a verifiable, objective answer in the future
4. **Specific**: Well-defined and unambiguous
5. **Relevant**: Related to the topic

### Example Valid Questions:
- "Will SpaceX land humans on Mars by December 31, 2030?" (specific future date)
- "Will US inflation fall below 2% before January 1, 2027?" (near-term, specific)
- "Will China reduce emissions by 5% before January 1, 2027?" (uses reasonable near-term date when article doesn't specify)
- "Will the EU AI Act be fully implemented by December 2025?" (uses date from article)

### Example Invalid Questions:
- "Will X happen by end of 2024?" (date is in the past - as of late 2025)
- "Will the Biden administration achieve X by 2030?" (references outdated context if Biden is no longer president)
- "Will the 2024 election be decided by November 2024?" (refers to past event)
- "Is climate change real?" (subjective, currently debated, no future date)
- "Should we regulate AI?" (subjective opinion, not yes/no with future answer)
- "What will happen to Bitcoin?" (not yes/no format)
- "Is AI dangerous?" (too vague, subjective, no specific date)
- "Will policy be implemented by March 2025?" (may be in past depending on current date)

## Notes

- **Web Search Enabled**: The default model uses `:online` suffix for real-time web access (costs $4 per 1,000 web results)
- **Recent News Focus**: Questions are based on news from the last `num_days` days (default: 7)
- **Future Dates Only**: All resolution dates are guaranteed to be in the future (after today's date)
- **Context Validation**: By default, questions are validated to ensure they make sense in today's context (e.g., removes questions about "Biden administration" if he's no longer president)
- **Date Preferences**: When articles specify dates, those are used; otherwise reasonable near-term dates (1-3 years) are chosen
- The function may return fewer than `n_topics` if the LLM cannot generate enough valid topics with questions
- Questions are automatically limited to at most `k_questions` per topic
- Topics with no valid questions are automatically filtered out
- Results will vary over time as news changes
- Validation adds one additional API call but significantly improves question quality

## Advanced Usage

### Using Different Models

```python
# Use Claude with web search (automatically adds :online)
result = generate_topics_and_questions(
    n_topics=5,
    k_questions=3,
    model="anthropic/claude-sonnet-4"  # Will become "anthropic/claude-sonnet-4:online"
)

# Use GPT-4o with web search (more capable but more expensive)
result = generate_topics_and_questions(
    n_topics=5,
    k_questions=3,
    model="openai/gpt-4o:online"
)

# Disable web search (not recommended - will not have access to recent news)
result = generate_topics_and_questions(
    n_topics=5,
    k_questions=3,
    model="openai/gpt-4o-mini",
    enable_web_search=False
)
```

### Error Handling

```python
try:
    result = generate_topics_and_questions(n_topics=5, k_questions=3)
except ValueError as e:
    print(f"Invalid parameters or missing API key: {e}")
except Exception as e:
    print(f"API call failed: {e}")
```

## License

This module is part of the eterniscollab project.
