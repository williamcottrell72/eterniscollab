# eterniscollab

A forecasting and LLM analysis toolkit with support for Polymarket data analysis, probability distribution estimation, and topic generation.

## Overview

**eterniscollab** provides tools for:
- **Polymarket Data**: Download and analyze historical prediction market data
- **Probability Estimation**: Generate probability distributions from LLM responses
- **Buzz Analysis**: Calculate interest × divisiveness scores for topics
- **Topic Generation**: Generate forecasting questions from trending news
- **LLM Integration**: Support for OpenAI, Claude, and OpenRouter API

## Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

Optional (for legacy direct API access):
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Quick Start

### Polymarket Data Download

```python
from polymarket_data import download_polymarket_prices_by_slug
from datetime import datetime

# Download price data for a market
df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    outcome_index=0,  # 0 = "Yes", 1 = "No"
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 2, 1),
    fidelity=10  # 10-minute intervals
)

print(df.head())
```

### Probability Distribution Estimation

```python
from probability_estimator import get_probability_distribution
import asyncio

async def estimate():
    result = await get_probability_distribution(
        prompt="Will it rain tomorrow in San Francisco?",
        n_samples=20,
        model="openai/gpt-4o-mini"
    )
    print(f"Mean: {result['mean']:.2f}")
    print(f"Std: {result['std']:.2f}")

asyncio.run(estimate())
```

### Topic Generation

```python
from topic_generator import generate_topics_and_questions

# Generate trending topics with forecasting questions
result = generate_topics_and_questions(
    n_topics=5,
    k_questions=3,
    model="openai/gpt-4o-mini:online"  # :online enables web search
)

for topic in result['topics']:
    print(f"\n{topic['title']}")
    for q in topic['questions']:
        print(f"  - {q['question']}")
```

## Main Functions

### Polymarket Data (`polymarket_data.py`)

#### `download_polymarket_prices(token_id, start_date, end_date, fidelity=1)`
Download historical price data for a specific token ID.
- **token_id**: Large numeric string identifying a market outcome
- **start_date/end_date**: datetime objects defining the range
- **fidelity**: Time resolution in minutes (1=minute, 10=10-min, 60=hourly, 1440=daily)
- **Returns**: DataFrame with columns: timestamp, price, unix_timestamp
- **Note**: Automatically chunks requests >14 days due to API limits

#### `download_polymarket_prices_by_slug(market_slug, outcome_index, start_date, end_date, fidelity=1)`
Convenience wrapper to download by market slug instead of token ID.
- **market_slug**: Market identifier (e.g., "fed-rate-hike-in-2025")
- **outcome_index**: 0 for first outcome (usually "Yes"), 1 for second (usually "No")
- **Returns**: DataFrame with price history

#### `get_event_markets(event_slug)`
Get all markets within an event with metadata.
- **event_slug**: Event identifier (e.g., "presidential-election-winner-2024")
- **Returns**: Dict mapping market names to metadata (token_ids, question, volume, etc.)

#### `download_polymarket_prices_by_event(event_slug, market_id, outcome_index=0, start_date, end_date, fidelity=1)`
Download data for a specific market within an event.
- **event_slug**: Event identifier
- **market_id**: Market name from get_event_markets() (e.g., "Donald Trump")
- **Returns**: DataFrame with price history

#### `slug_to_token_ids(slug, is_event=False)`
Map a slug to token IDs.
- **slug**: Market or event slug
- **is_event**: True for events, False for individual markets
- **Returns**: Dict mapping market names to token ID lists

#### `get_all_closed_markets(cache_dir="data/polymarket", overwrite=False, max_markets=None)`
Fetch metadata for all closed Polymarket markets.
- **Returns**: DataFrame with market metadata (slug, question, volume, end_date, etc.)
- **Note**: Results are cached for fast reloading

### Probability Estimation (`probability_estimator.py`)

#### `get_probability_distribution(prompt, n_samples=20, reword_temperature=0.7, prompt_temperature=0.5, model="openai/gpt-4o-mini")`
Estimate probability distribution by querying LLM multiple times.
- **prompt**: Question to estimate probability for
- **n_samples**: Number of probability samples to collect
- **reword_temperature**: Controls prompt variation (0=no rewording, 1=high variation)
- **prompt_temperature**: Controls LLM response randomness
- **model**: OpenRouter model identifier
- **Returns**: Dict with mean, std, probabilities, and reworded_prompts

#### `reword_prompt(original_prompt, temperature=0.7, model="openai/gpt-4o-mini")`
Reword a prompt while preserving meaning.
- **temperature**: Controls variation level (0=minimal, 1=high)
- **Returns**: Reworded version of the prompt

#### `query_probability(prompt, temperature=0.5, model="openai/gpt-4o-mini")`
Get single probability estimate (0-1) from LLM.
- **Returns**: Float between 0 and 1

### Buzz Analysis (`buzz.py`)

#### `get_buzz_score(topic, n_samples=10, temperature=0.5, model="openai/gpt-4o-mini")`
Calculate buzz score (interest × divisiveness) for a topic.
- **topic**: Topic to analyze
- **n_samples**: Number of samples to average
- **Returns**: Dict with interest, divisiveness, and buzz scores (all 0-1)

#### `query_interest(topic, temperature=0.5, model="openai/gpt-4o-mini")`
Get interest score (0-1) for a topic.
- **Returns**: Float between 0 and 1

#### `query_divisiveness(topic, temperature=0.5, model="openai/gpt-4o-mini")`
Get divisiveness score (0-1) for a topic.
- **Returns**: Float between 0 and 1

#### `estimate_daily_volume(query, n_examples=20, model="openai/gpt-4o-mini", print_prompt=False)`
Estimate average daily trading volume for a prediction market question using historical data.
- **query**: The prediction market question to estimate
- **n_examples**: Number of historical examples to use (default: 20)
- **model**: OpenRouter model identifier
- **print_prompt**: If True, display the full prompt with examples
- **Returns**: Dict with estimated_daily_volume (USD), n_examples, model, and examples list
- **Note**: Uses closed markets data to provide context-aware estimates

### Topic Generation (`topic_generator.py`)

#### `generate_topics_and_questions(n_topics=5, k_questions=3, model="openai/gpt-4o-mini:online")`
Generate trending topics and forecasting questions.
- **n_topics**: Number of trending topics to generate
- **k_questions**: Maximum questions per topic (actual number may be less)
- **model**: OpenRouter model (use `:online` suffix for web search)
- **Returns**: Dict with topics list, each containing title, summary, and questions

### Utilities (`utils.py`)

#### `query_llm_for_numeric_value(prompt, temperature=0.5, model="openai/gpt-4o-mini")`
Query LLM and extract numeric value (0-1).
- **Returns**: Float between 0 and 1

#### `query_llm_for_text(prompt, temperature=0.5, model="openai/gpt-4o-mini")`
Get raw text response from LLM.
- **Returns**: String response

#### `extract_numeric_value(text)`
Parse numeric value from text (handles decimals, percentages, fractions).
- **Returns**: Float between 0 and 1, or None if no value found

## Project Structure

```
eterniscollab/
├── Core Library Modules
│   ├── polymarket_data.py          # Polymarket data downloader
│   ├── probability_estimator.py    # Probability distribution estimation
│   ├── buzz.py                     # Buzz score analysis
│   ├── topic_generator.py          # Topic and question generation
│   ├── utils.py                    # Shared utilities
│   └── generate_topic_rankings.py  # Generate and rank topics
│
├── scripts/                        # Command-line scripts
│   ├── Example scripts for all modules
│   ├── Utility scripts (verify, fix issues)
│   └── Bulk download scripts
│
├── docs/                           # Documentation
│   ├── INDEX.md                    # Documentation index
│   ├── POLYMARKET_README.md        # Polymarket API guide
│   ├── QUICK_START_POLYMARKET.md   # Quick reference
│   ├── API_LIMITS_EXPLANATION.md   # 15-day limit and chunking
│   ├── TOPIC_GENERATOR_README.md   # Topic generator guide
│   └── ...                         # Additional docs
│
├── notebooks/                      # Jupyter notebooks
│   ├── probability_distribution_analysis.ipynb
│   ├── buzz_distribution_analysis.ipynb
│   └── polymarket_exploration.ipynb
│
├── tests/                          # Unit tests
│   ├── test_polymarket_download.py
│   ├── test_topic_generator.py
│   └── ...                         # Additional tests
│
├── data/                           # Cached data (git-ignored)
│   └── polymarket/                 # Polymarket price data cache
│
├── README.md                       # This file
├── CLAUDE.md                       # Project instructions for Claude Code
└── requirements.txt                # Python dependencies
```

## Key Features

### Polymarket Integration
- Download historical price data with minute-level resolution
- Automatic chunking for date ranges >14 days
- Caching to avoid redundant API calls
- Support for events (multiple related markets)
- Direct slug queries for fast market lookup

### LLM Analysis
- Probability distribution estimation via prompt rewording
- Buzz score calculation (interest × divisiveness)
- Support for multiple LLM providers via OpenRouter
- Temperature control for both prompts and responses

### Topic Generation
- Real-time web search for trending topics (via `:online` models)
- Generate yes/no forecasting questions with resolution dates
- Enforce topic diversity and question quality
- JSON-structured output for easy parsing

## Documentation

Detailed documentation is available in the `docs/` folder:

- **[Polymarket Guide](docs/POLYMARKET_README.md)**: Complete Polymarket API documentation
- **[Quick Start](docs/QUICK_START_POLYMARKET.md)**: Quick reference for common tasks
- **[API Limits](docs/API_LIMITS_EXPLANATION.md)**: Understanding the 15-day limit
- **[Fidelity Parameter](docs/FIDELITY_EXPLANATION.md)**: Time resolution explained
- **[Topic Generator](docs/TOPIC_GENERATOR_README.md)**: Topic generation guide
- **[OpenRouter Setup](docs/OPENROUTER_SETUP.md)**: API configuration

## Examples

### Example 1: Presidential Election Analysis

```python
from polymarket_data import get_event_markets, download_polymarket_prices_by_event
from datetime import datetime

# Discover all candidate markets
markets = get_event_markets("presidential-election-winner-2024")
print("Available candidates:", list(markets.keys()))

# Download Trump's odds for October 2024
df = download_polymarket_prices_by_event(
    event_slug="presidential-election-winner-2024",
    market_id="Donald Trump",
    outcome_index=0,  # "Yes" token
    start_date=datetime(2024, 10, 1),
    end_date=datetime(2024, 11, 1),
    fidelity=60  # Hourly data
)

print(f"Downloaded {len(df)} data points")
print(f"Price range: {df['price'].min():.3f} - {df['price'].max():.3f}")
```

### Example 2: Probability Distribution Analysis

```python
from probability_estimator import get_probability_distribution
import asyncio

async def analyze():
    # Get distribution with varying prompt wordings
    result = await get_probability_distribution(
        prompt="Will Bitcoin exceed $100k by end of 2025?",
        n_samples=50,
        reword_temperature=0.8,  # High variation in rewording
        model="anthropic/claude-sonnet-4"
    )

    print(f"Mean probability: {result['mean']:.1%}")
    print(f"Std deviation: {result['std']:.3f}")
    print(f"Range: {min(result['probabilities']):.1%} - {max(result['probabilities']):.1%}")

asyncio.run(analyze())
```

### Example 3: Buzz Score Analysis

```python
from buzz import get_buzz_score

topics = [
    "Artificial Intelligence regulation",
    "Federal Reserve interest rates",
    "Celebrity gossip"
]

for topic in topics:
    scores = get_buzz_score(topic, n_samples=20)
    print(f"\n{topic}:")
    print(f"  Interest: {scores['interest']:.2f}")
    print(f"  Divisiveness: {scores['divisiveness']:.2f}")
    print(f"  Buzz: {scores['buzz']:.2f}")
```

### Example 4: Estimate Market Volume

```python
from buzz import estimate_daily_volume

# Estimate trading volume for a new market question
result = estimate_daily_volume(
    query="Will Bitcoin exceed $100k by end of 2025?",
    n_examples=20,  # Use 20 historical markets as context
    print_prompt=False  # Set to True to see the full prompt
)

print(f"Estimated Daily Volume: ${result['estimated_daily_volume']:.2f}")
print(f"Based on {result['n_examples']} historical examples")

# The function uses closed markets with varying volumes to provide context
# Estimates consider topic relevance, time horizon, and similar historical markets
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_topic_generator.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Jupyter Notebooks

Launch notebooks for interactive analysis:

```bash
jupyter notebook notebooks/
```

Available notebooks:
- `probability_distribution_analysis.ipynb`: Analyze LLM probability distributions
- `buzz_distribution_analysis.ipynb`: Analyze interest and divisiveness scores

## Contributing

When adding new features:
1. Add type hints and docstrings
2. Write unit tests
3. Update relevant documentation in `docs/`
4. Ensure notebooks remain runnable from start to finish

## License

MIT License

## Support

For issues or questions, please check the documentation in the `docs/` folder or open an issue on GitHub.
