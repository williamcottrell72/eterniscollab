# Market Pipeline Documentation

## Overview

The Market Pipeline is an end-to-end system for generating prediction market questions from recent news, estimating their trading metrics, and allocating capital across them.

**Module**: `market_pipeline.py`

## Pipeline Steps

The pipeline executes five sequential steps:

### 1. Question Generation
- **Function**: `generate_questions()`
- **Purpose**: Generate prediction market questions from recent news
- **Uses**: `topic_generator.generate_topics_and_questions()`
- **Input**: Number of topics and questions per topic
- **Output**: List of questions with resolution dates and criteria

### 2. Volume Estimation
- **Function**: `estimate_volumes()`
- **Purpose**: Estimate average daily trading volume for each question
- **Uses**: `buzz.estimate_daily_volume()`
- **Input**: List of questions
- **Output**: Questions enriched with `estimated_daily_volume`

### 3. Probability Estimation
- **Function**: `estimate_probabilities()`
- **Purpose**: Estimate probability of "Yes" resolution
- **Uses**: `probability_estimator.get_probability_estimate()`
- **Input**: List of questions with volume estimates
- **Output**: Questions enriched with probability statistics (mean, median, std)

### 4. Capital Allocation
- **Function**: `allocate_capital()`
- **Purpose**: Allocate capital proportional to estimated volume
- **Input**: List of questions with volume and probability estimates, total capital
- **Output**: Questions enriched with `allocated_capital`
- **Logic**: Capital is allocated proportional to estimated daily volume:
  ```
  allocation[i] = (volume[i] / sum(volumes)) * total_capital
  ```

### 5. Results Saving
- **Function**: `save_results()`
- **Purpose**: Save results with full reproducibility metadata
- **Input**: Enriched questions, pipeline configuration
- **Output**: Directory with timestamped results
- **Files Created**:
  - `results.csv`: Main results in tabular format
  - `results.json`: Full results including nested data
  - `pipeline_config.json`: Complete pipeline configuration
  - `summary.json`: Summary statistics

## Usage

### Basic Usage

```python
from market_pipeline import run_market_pipeline

results = run_market_pipeline(
    n_topics=5,
    k_questions_per_topic=2,
    total_capital=10000.0
)

print(f"Generated {len(results['questions'])} questions")
print(f"Results saved to: {results['output_dir']}")
```

### Advanced Usage

```python
results = run_market_pipeline(
    n_topics=10,
    k_questions_per_topic=3,
    total_capital=50000.0,
    n_volume_examples=30,
    n_probability_samples=20,
    question_model="openai/gpt-4o:online",
    volume_model="anthropic/claude-sonnet-4",
    probability_model="openai/gpt-4o-mini",
    reword_temperature=0.8,
)
```

### Command-Line Usage

```bash
# Run with default settings ($10,000 capital)
python market_pipeline.py

# Run with custom capital allocation
python market_pipeline.py 50000

# Run test pipeline
python scripts/test_market_pipeline.py
```

## Parameters

### `run_market_pipeline()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_topics` | int | 5 | Number of topics to generate |
| `k_questions_per_topic` | int | 2 | Max questions per topic |
| `total_capital` | float | 10000.0 | Total capital to allocate (USD) |
| `n_volume_examples` | int | 20 | Historical examples for volume estimation |
| `n_probability_samples` | int | 10 | Samples for probability estimation |
| `question_model` | str | "openai/gpt-4o-mini:online" | Model for question generation (must have :online) |
| `volume_model` | str | "openai/gpt-4o-mini" | Model for volume estimation |
| `probability_model` | str | "openai/gpt-4o-mini" | Model for probability estimation |
| `reword_temperature` | float | 0.7 | Temperature for prompt rewording |
| `api_key` | str | None | OpenRouter API key (or use env var) |
| `output_dir` | str | "data/pipelines" | Base directory for outputs |

## Output Structure

Each pipeline run creates a timestamped directory:

```
data/pipelines/
└── YYYYMMDD_HHMM/
    ├── results.csv           # Main results (CSV format)
    ├── results.json          # Full results (JSON format)
    ├── pipeline_config.json  # Configuration for reproducibility
    └── summary.json          # Summary statistics
```

### `results.csv` Columns

- `topic`: Topic name
- `question`: Prediction market question
- `resolution_date`: When the question resolves (currently None, could be extracted in future)
- `resolution_criteria`: How to determine the outcome (currently None, could be extracted in future)
- `estimated_daily_volume`: Estimated daily trading volume (USD)
- `probability_mean`: Mean probability of "Yes" resolution
- `probability_median`: Median probability
- `probability_std`: Standard deviation of probability samples
- `allocated_capital`: Capital allocated to this market (USD)
- `allocation_method`: Method used for allocation
- `volume_fraction`: Fraction of total volume

**Note**: `resolution_date` and `resolution_criteria` are currently set to None because `generate_topics_and_questions()` returns simple question strings. These could be extracted from question text or added as a future enhancement.

### `pipeline_config.json` Contents

Complete configuration to reproduce the run:
- All function parameters
- Model identifiers
- Timestamp
- Output directory

### `summary.json` Contents

High-level statistics:
- Number of questions generated
- Number with successful volume estimates
- Number with successful probability estimates
- Total capital allocated
- Average probability and volume

## Error Handling

The pipeline is designed to be robust:

1. **Volume Estimation Failures**: If volume estimation fails for a question, it's marked with `volume_error` and receives $0 allocation
2. **Probability Estimation Failures**: If probability estimation fails, it's marked with `probability_error` but still receives capital allocation based on volume
3. **Partial Failures**: The pipeline continues even if some questions fail
4. **Full Failures**: If a critical step fails (e.g., question generation), the entire pipeline stops with an error message

## Reproducibility

To reproduce a pipeline run:

1. Navigate to the run directory: `data/pipelines/YYYYMMDD_HHMM/`
2. Load the configuration: `cat pipeline_config.json`
3. Re-run with the same parameters:

```python
import json
from market_pipeline import run_market_pipeline

with open('data/pipelines/20250116_1430/pipeline_config.json') as f:
    config = json.load(f)

results = run_market_pipeline(**config)
```

**Note**: Results will differ slightly due to:
- New news content (for question generation with :online models)
- LLM sampling randomness
- Updated closed markets data (for volume estimation)

## Dependencies

- `topic_generator.py`: Question generation
- `buzz.py`: Volume estimation
- `probability_estimator.py`: Probability estimation
- `polymarket_data.py`: Historical market data (via buzz.py)
- OpenRouter API access

## Environment Setup

```bash
# Required
export OPENROUTER_API_KEY="your-key-here"

# Optional (for reproducibility testing)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## Examples

### Example 1: Conservative Run

Small number of questions, thorough estimation:

```python
results = run_market_pipeline(
    n_topics=3,
    k_questions_per_topic=1,
    total_capital=5000.0,
    n_volume_examples=30,
    n_probability_samples=20,
)
```

### Example 2: Large-Scale Run

Many questions, fast estimation:

```python
results = run_market_pipeline(
    n_topics=20,
    k_questions_per_topic=5,
    total_capital=100000.0,
    n_volume_examples=10,
    n_probability_samples=5,
)
```

### Example 3: High-Quality Models

Use premium models for better estimates:

```python
results = run_market_pipeline(
    n_topics=10,
    k_questions_per_topic=3,
    total_capital=50000.0,
    question_model="openai/gpt-4o:online",
    volume_model="anthropic/claude-sonnet-4",
    probability_model="openai/gpt-4o",
)
```

## Cost Estimation

Approximate OpenRouter costs per pipeline run:

- Question generation: ~$0.01-0.05 per question (with :online, includes $4/1000 searches)
- Volume estimation: ~$0.001-0.005 per question
- Probability estimation: ~$0.01-0.05 per question (depends on n_samples)

**Example**: 10 questions with default settings ≈ $0.20-1.00

Costs vary significantly by:
- Model choice (gpt-4o vs gpt-4o-mini)
- Number of samples/examples
- Web search usage (`:online` models)

## Troubleshooting

### "OPENROUTER_API_KEY not set"
Set the environment variable:
```bash
export OPENROUTER_API_KEY="your-key-here"
```

### "No markets with valid daily_volume found"
The closed markets cache may be empty. Run:
```python
from polymarket_data import get_all_closed_markets
get_all_closed_markets(overwrite=True)
```

### Volume estimation taking too long
Reduce `n_volume_examples` (e.g., from 20 to 10)

### Probability estimation failing
Check that questions are properly formatted as yes/no questions with clear resolution criteria

## Future Enhancements

Possible improvements:
- Alternative allocation strategies (e.g., Kelly criterion, equal allocation)
- Risk-adjusted allocation (based on probability uncertainty)
- Multi-objective optimization (volume + probability + other factors)
- Incremental pipeline execution (save after each step)
- Parallel processing for faster execution
- Portfolio optimization across questions
- Backtesting framework using historical data

## Integration

The pipeline can be integrated into larger systems:

```python
# Schedule daily runs
from market_pipeline import run_market_pipeline
import schedule

def daily_pipeline():
    results = run_market_pipeline(
        n_topics=10,
        k_questions_per_topic=2,
        total_capital=10000.0
    )
    # Process results...

schedule.every().day.at("09:00").do(daily_pipeline)
```

## See Also

- [Topic Generator README](TOPIC_GENERATOR_README.md)
- [Polymarket Data README](POLYMARKET_README.md)
- [Probability Estimator Documentation](../probability_estimator.py)
- [Buzz Analyzer Documentation](../buzz.py)
