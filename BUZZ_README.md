# Buzz Analyzer

A tool for measuring the "buzz" around topics using LLMs (OpenAI and Claude) to quantify public interest and divisiveness.

## Overview

The Buzz Analyzer provides two key metrics:

1. **Interest Level (0-1)**: Quantifies raw public attention based on:
   - Internet traffic and search volume
   - News coverage frequency
   - Social media trends and mentions

2. **Divisiveness (0-1)**: Measures how polarizing a topic is based on:
   - Emotional intensity in discussions
   - Presence of opposing viewpoints
   - Anger levels and extreme rhetoric
   - Formation of tribal camps

3. **Buzz Score (0-1)**: Combined metric calculated as `interest × divisiveness`
   - High buzz requires both high interest AND high divisiveness
   - Example: A topic everyone agrees on = high interest, low divisiveness = moderate buzz
   - Example: A controversial topic no one cares about = low interest, high divisiveness = low buzz

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Using OpenAI
python buzz.py openai YOUR_API_KEY "Climate Change"

# Using Claude
python buzz.py claude YOUR_API_KEY "Artificial Intelligence"

# With environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
python buzz.py openai _ "2024 US Presidential Election"
```

### Python API

```python
from buzz import get_buzz_score

# Get full buzz analysis
result = get_buzz_score(
    topic="Climate Change",
    provider="claude",  # or "openai"
    api_key=None,  # Uses environment variable if None
    temperature=0.5
)

print(f"Interest: {result['interest']:.3f}")
print(f"Divisiveness: {result['divisiveness']:.3f}")
print(f"Buzz: {result['buzz']:.3f}")
```

### Individual Metrics

```python
from buzz import (
    query_claude_interest,
    query_claude_divisiveness,
    query_openai_interest,
    query_openai_divisiveness
)

# Get just interest level
interest = query_claude_interest(
    topic="Quantum Computing",
    api_key="your-key"
)

# Get just divisiveness
divisiveness = query_openai_divisiveness(
    topic="Gun Control",
    api_key="your-key"
)
```

## Example Results

Based on actual test runs:

| Topic | Interest | Divisiveness | Buzz | Interpretation |
|-------|----------|--------------|------|----------------|
| 2024 US Presidential Election | 0.85 | 0.88 | 0.75 | 🔥 Very high buzz |
| Artificial Intelligence | 0.92 | 0.50 | 0.46 | 📊 Moderate buzz (high interest, moderate controversy) |
| The Beatles | 0.60 | 0.15 | 0.09 | 📉 Low buzz (interest without controversy) |
| Medieval Latin Poetry | 0.15 | 0.05 | 0.01 | 📉 Very low buzz |

## Prompts

The system uses carefully crafted prompts to ensure consistent scoring:

### Interest Level Prompt
```
Rate the current level of public interest in this topic on a scale from 0 to 1, where:
- 0 = Virtually no public interest, minimal to no internet traffic or news coverage
- 0.25 = Low interest, occasional mentions in niche sources
- 0.5 = Moderate interest, regular news coverage and social media discussion
- 0.75 = High interest, frequent major news coverage and trending online
- 1.0 = Extremely high interest, dominating news cycles and viral across all platforms

Consider recent internet traffic, news coverage, social media trends, and search volume.
Provide ONLY a numeric score between 0 and 1, with no additional explanation.
```

### Divisiveness Prompt
```
Rate how divisive and polarizing this topic is on a scale from 0 to 1, where:
- 0 = Not divisive at all, broad consensus and calm discussion
- 0.25 = Slightly divisive, some disagreement but mostly civil discourse
- 0.5 = Moderately divisive, clear opposing viewpoints with passionate debate
- 0.75 = Highly divisive, strong emotions, heated arguments, and tribal camps
- 1.0 = Extremely divisive, intense anger, extreme rhetoric, and deep polarization

Consider the intensity of emotions, presence of opposing camps, anger levels, and extremity of rhetoric.
Provide ONLY a numeric score between 0 and 1, with no additional explanation.
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/test_buzz.py -v

# Specific test classes
pytest tests/test_buzz.py::TestClaudeInterest -v
pytest tests/test_buzz.py::TestClaudeDivisiveness -v
pytest tests/test_buzz.py::TestBuzzScore -v

# With output
pytest tests/test_buzz.py -v -s
```

## API Functions

### Core Functions

- `get_buzz_score(topic, provider, api_key, model, temperature)`: Get all metrics at once
- `query_openai_interest(topic, api_key, model, temperature)`: OpenAI interest score
- `query_claude_interest(topic, api_key, model, temperature)`: Claude interest score
- `query_openai_divisiveness(topic, api_key, model, temperature)`: OpenAI divisiveness score
- `query_claude_divisiveness(topic, api_key, model, temperature)`: Claude divisiveness score

### Helper Functions

- `_extract_score(text)`: Extract numeric score from LLM response text

## Design Decisions

1. **Separate Metrics**: Interest and divisiveness are measured independently to provide more nuanced understanding
2. **Multiplicative Buzz Score**: Using multiplication ensures both dimensions matter (not just sum)
3. **Clear Scale Definitions**: Prompts provide specific anchor points at 0, 0.25, 0.5, 0.75, and 1.0
4. **Minimal Output**: Requesting only numeric scores reduces parsing errors
5. **Temperature Control**: Default 0.5 balances consistency with reasonable variation

## Future Enhancements

Potential additions:
- Time-series tracking of buzz over time
- Sentiment analysis (positive vs negative buzz)
- Geographic variation in buzz
- Comparison to historical baselines
- Multi-sample averaging for more robust scores
- Confidence intervals

## License

Part of the eterniscollab project.
