# Probability Evolution Over Time

Track how LLM probability estimates change with different knowledge cutoff dates.

## Overview

The `get_probability_distribution_over_time()` function allows you to see how forecasts evolve as new information becomes available. This is useful for:

- Understanding how forecasts improve over time
- Measuring forecast calibration at different time points
- Analyzing the impact of new information on predictions
- Studying LLM temporal reasoning capabilities

## Quick Start

```python
import asyncio
from datetime import datetime
from probability_estimator import get_probability_distribution_over_time
import numpy as np

async def main():
    # Track how election forecasts evolved from January to November 2024
    result = await get_probability_distribution_over_time(
        prompt="Will Donald Trump win the 2024 US Presidential election?",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 11, 5),
        frequency_days=30,  # Monthly updates
        n_samples=10
    )

    # Display results
    for date_str, dist in sorted(result.items()):
        mean = np.mean(dist['probabilities'])
        print(f"{date_str}: {mean:.1%}")

asyncio.run(main())
```

## Function Signature

```python
async def get_probability_distribution_over_time(
    prompt: str,
    start_date: datetime,
    end_date: datetime,
    frequency_days: int = 7,
    **kwargs
) -> Dict[str, Dict[str, Any]]
```

### Parameters

- **prompt** (str): The forecasting question to ask
- **start_date** (datetime): First knowledge cutoff date
- **end_date** (datetime): Last knowledge cutoff date
- **frequency_days** (int): Days between queries (default: 7)
- **kwargs**: Additional arguments passed to `get_probability_distribution()`:
  - `n_samples` (int): Number of samples per distribution (default: 10)
  - `reword_temperature` (float): Prompt rewording flexibility (default: 0.5)
  - `prompt_temperature` (float): Sampling temperature (default: 0.7)
  - `model` (str): OpenRouter model identifier (default: "openai/gpt-4o-mini")
  - `api_key` (str): OpenRouter API key (default: from OPENROUTER_API_KEY env var)

### Returns

Dictionary mapping date strings (YYYY-MM-DD) to distribution results:

```python
{
    "2024-01-01": {
        "probabilities": [0.45, 0.48, 0.42, ...],
        "reworded_prompts": ["Will Trump win...", ...],
        "model": "openai/gpt-4o-mini",
        "n_samples": 10,
        "reword_temperature": 0.5,
        "prompt_temperature": 0.7,
        "knowledge_cutoff_date": "January 01, 2024"
    },
    "2024-02-01": { ... },
    ...
}
```

## Date Sampling Logic

The function always includes:
1. **Start date** - always included
2. **Intermediate dates** - spaced by `frequency_days`
3. **End date** - always included

### Examples

**Example 1:** `start=2024-01-01, end=2024-01-10, frequency=7`
- Dates queried: `2024-01-01`, `2024-01-08`, `2024-01-10`

**Example 2:** `start=2024-01-01, end=2024-01-31, frequency=7`
- Dates queried: `2024-01-01`, `2024-01-08`, `2024-01-15`, `2024-01-22`, `2024-01-29`, `2024-01-31`

**Example 3:** `start=2024-01-01, end=2024-12-31, frequency=30`
- Dates queried: Monthly intervals from Jan 1 to Dec 31

## Usage Examples

### Example 1: Monthly Election Forecast Evolution

Track how forecasts changed monthly throughout 2024:

```python
import asyncio
from datetime import datetime
from probability_estimator import get_probability_distribution_over_time
import numpy as np

async def track_election():
    result = await get_probability_distribution_over_time(
        prompt="Will Donald Trump win the 2024 US Presidential election?",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 11, 5),
        frequency_days=30,
        n_samples=20,
        model="openai/gpt-4o-mini"
    )

    for date_str, dist in sorted(result.items()):
        probs = dist['probabilities']
        print(f"{date_str}: {np.mean(probs):.1%} ± {np.std(probs):.1%}")

asyncio.run(track_election())
```

Output:
```
2024-01-01: 48.5% ± 5.2%
2024-01-31: 51.2% ± 4.8%
2024-03-01: 53.7% ± 6.1%
...
2024-11-05: 62.3% ± 3.4%
```

### Example 2: Weekly Policy Forecast

Track weekly changes for a policy question:

```python
async def track_policy():
    result = await get_probability_distribution_over_time(
        prompt="Will the EU implement the AI Act by December 31, 2025?",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 31),
        frequency_days=7,
        n_samples=15,
        reword_temperature=0.7,
        model="openai/gpt-4o-mini"
    )

    return result

asyncio.run(track_policy())
```

### Example 3: Compare Multiple Models

Track the same question with different models:

```python
async def compare_models():
    prompt = "Will SpaceX land humans on Mars by 2030?"
    dates = (datetime(2024, 1, 1), datetime(2024, 6, 1))

    models = ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4"]

    results = {}
    for model in models:
        results[model] = await get_probability_distribution_over_time(
            prompt=prompt,
            start_date=dates[0],
            end_date=dates[1],
            frequency_days=30,
            n_samples=10,
            model=model
        )

    # Compare results
    for model, result in results.items():
        print(f"\n{model}:")
        for date_str, dist in sorted(result.items()):
            mean = np.mean(dist['probabilities'])
            print(f"  {date_str}: {mean:.1%}")

asyncio.run(compare_models())
```

### Example 4: Analyze with Statistics

Use the helper function to analyze evolution:

```python
from probability_estimator import (
    get_probability_distribution_over_time,
    analyze_probability_evolution
)

async def analyze_evolution():
    # Get time series data
    result = await get_probability_distribution_over_time(
        prompt="Will the Fed raise rates by Q4 2025?",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        frequency_days=30,
        n_samples=20
    )

    # Analyze evolution
    stats_df = await analyze_probability_evolution(result)

    print(stats_df)

    # Plot results
    import plotly.express as px
    fig = px.line(
        stats_df,
        x='date',
        y='mean',
        error_y='std',
        title='Probability Evolution Over Time',
        labels={'mean': 'Mean Probability', 'date': 'Knowledge Cutoff Date'}
    )
    fig.show()

asyncio.run(analyze_evolution())
```

## Analysis Functions

### `analyze_probability_evolution()`

Helper function to compute statistics from time series results:

```python
async def analyze_probability_evolution(
    time_series_results: Dict[str, Dict[str, Any]],
    return_dataframe: bool = True
) -> pd.DataFrame | List[Dict]
```

Returns a DataFrame with columns:
- `date`: Knowledge cutoff date
- `mean`: Mean probability
- `median`: Median probability
- `std`: Standard deviation
- `min`: Minimum probability
- `max`: Maximum probability
- `q25`: 25th percentile
- `q75`: 75th percentile
- `n_samples`: Number of samples
- `model`: Model used
- `knowledge_cutoff_date`: Formatted cutoff date

## Common Use Cases

### 1. Measuring Forecast Improvement

See how forecasts converge to the correct answer as more information becomes available:

```python
# Track forecasts leading up to a known outcome
result = await get_probability_distribution_over_time(
    prompt="Will Trump win the 2024 election?",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 5),  # Election day
    frequency_days=14
)

# Compare early vs late forecasts
early_mean = np.mean(result['2024-01-01']['probabilities'])
late_mean = np.mean(result['2024-11-05']['probabilities'])
# Actual outcome: Trump won (1.0)

print(f"Early forecast error: {abs(early_mean - 1.0):.2f}")
print(f"Late forecast error: {abs(late_mean - 1.0):.2f}")
print(f"Improvement: {abs(early_mean - 1.0) - abs(late_mean - 1.0):.2f}")
```

### 2. Studying Information Shocks

Identify when major news events caused forecast changes:

```python
result = await get_probability_distribution_over_time(
    prompt="Will Biden win the 2024 election?",
    start_date=datetime(2024, 6, 1),
    end_date=datetime(2024, 8, 1),
    frequency_days=7
)

# Calculate week-over-week changes
dates = sorted(result.keys())
for i in range(1, len(dates)):
    prev_mean = np.mean(result[dates[i-1]]['probabilities'])
    curr_mean = np.mean(result[dates[i]]['probabilities'])
    change = curr_mean - prev_mean

    if abs(change) > 0.1:  # 10% change threshold
        print(f"Large change on {dates[i]}: {change:+.1%}")
        # Biden dropped out on July 21, 2024
```

### 3. Calibration Analysis

Measure how well-calibrated forecasts are at different time horizons:

```python
# For resolved questions, compare forecasts to outcomes
questions = [
    ("Will Trump win 2024?", datetime(2024, 11, 5), 1.0),  # Actual outcome
    ("Will Biden drop out?", datetime(2024, 7, 21), 1.0),
    # ... more questions
]

for question, resolution_date, outcome in questions:
    # Get forecasts at different time horizons
    result = await get_probability_distribution_over_time(
        prompt=question,
        start_date=resolution_date - timedelta(days=180),  # 6 months before
        end_date=resolution_date,
        frequency_days=30
    )

    # Calculate calibration for each time point
    for date_str, dist in result.items():
        mean_forecast = np.mean(dist['probabilities'])
        brier_score = (mean_forecast - outcome) ** 2
        days_before = (resolution_date - datetime.strptime(date_str, '%Y-%m-%d')).days

        print(f"{days_before} days before: forecast={mean_forecast:.2f}, "
              f"brier={brier_score:.3f}")
```

### 4. Model Comparison Over Time

Compare how different models' forecasts evolve:

```python
models = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4"
]

question = "Will there be a recession in 2025?"
dates = (datetime(2024, 1, 1), datetime(2024, 12, 31))

all_results = {}
for model in models:
    all_results[model] = await get_probability_distribution_over_time(
        prompt=question,
        start_date=dates[0],
        end_date=dates[1],
        frequency_days=30,
        n_samples=15,
        model=model
    )

# Analyze model agreement over time
for date_str in sorted(all_results[models[0]].keys()):
    model_means = [
        np.mean(all_results[model][date_str]['probabilities'])
        for model in models
    ]
    agreement = np.std(model_means)  # Low std = high agreement
    print(f"{date_str}: agreement={1-agreement:.2f}, "
          f"means={[f'{m:.2f}' for m in model_means]}")
```

## Performance Considerations

### API Costs

Each call to `get_probability_distribution_over_time()` makes:
- **Number of API calls** = `n_dates × n_samples × 2`
  - `n_dates` = number of time points (start, intermediate, end)
  - `n_samples` = samples per distribution
  - `×2` = one for rewording, one for probability

**Example cost calculation:**
```python
# Monthly tracking for 1 year
n_dates = 12
n_samples = 10
total_calls = 12 × 10 × 2 = 240 API calls

# At $0.15 per 1M tokens with gpt-4o-mini:
# ~100 tokens per call × 240 = 24,000 tokens
# Cost: ~$0.004
```

### Execution Time

Approximate execution times:
- **1 date, 10 samples**: ~5-10 seconds
- **12 dates, 10 samples**: ~1-2 minutes
- **52 dates, 20 samples**: ~10-15 minutes

Tips for faster execution:
- Use `n_samples=5` for quick tests
- Use larger `frequency_days` (e.g., 30 instead of 7)
- Consider parallelizing across different questions (not dates)

## Integration with Other Tools

### Visualize with Plotly

```python
import plotly.graph_objects as go

async def plot_evolution():
    result = await get_probability_distribution_over_time(...)
    stats = await analyze_probability_evolution(result)

    fig = go.Figure()

    # Mean line with error band
    fig.add_trace(go.Scatter(
        x=stats['date'],
        y=stats['mean'],
        mode='lines+markers',
        name='Mean',
        line=dict(color='blue')
    ))

    # Confidence band (mean ± std)
    fig.add_trace(go.Scatter(
        x=stats['date'],
        y=stats['mean'] + stats['std'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=stats['date'],
        y=stats['mean'] - stats['std'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(0,100,200,0.2)',
        fill='tonexty',
        name='±1 Std Dev'
    ))

    fig.update_layout(
        title='Probability Evolution Over Time',
        xaxis_title='Knowledge Cutoff Date',
        yaxis_title='Probability',
        yaxis_range=[0, 1]
    )

    fig.show()
```

### Export to CSV

```python
async def export_results():
    result = await get_probability_distribution_over_time(...)
    stats = await analyze_probability_evolution(result)

    # Export statistics
    stats.to_csv('probability_evolution.csv', index=False)

    # Export raw probabilities
    import pandas as pd
    rows = []
    for date_str, dist in result.items():
        for i, prob in enumerate(dist['probabilities']):
            rows.append({
                'date': date_str,
                'sample': i,
                'probability': prob,
                'reworded_prompt': dist['reworded_prompts'][i]
            })

    df = pd.DataFrame(rows)
    df.to_csv('probability_samples.csv', index=False)
```

### Compare to Polymarket Data

```python
from polymarket_data import download_polymarket_prices

async def compare_to_market():
    # Get LLM forecasts
    llm_result = await get_probability_distribution_over_time(
        prompt="Will Trump win the 2024 election?",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 11, 5),
        frequency_days=7
    )

    # Get Polymarket data
    market_df = download_polymarket_prices(
        token_id="...",  # Trump token ID
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 11, 5),
        fidelity=1440  # Daily
    )

    # Compare
    for date_str, dist in llm_result.items():
        llm_mean = np.mean(dist['probabilities'])

        # Get closest market price
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        market_price = market_df[market_df['timestamp'].dt.date == date_obj.date()]['price'].mean()

        print(f"{date_str}: LLM={llm_mean:.2%}, Market={market_price:.2%}, "
              f"Diff={llm_mean-market_price:+.2%}")
```

## Troubleshooting

### Error: "start_date must be before end_date"

Make sure your dates are in chronological order:
```python
# Wrong
start_date=datetime(2024, 12, 31)
end_date=datetime(2024, 1, 1)

# Correct
start_date=datetime(2024, 1, 1)
end_date=datetime(2024, 12, 31)
```

### Error: "frequency_days must be at least 1"

Use a positive integer for frequency:
```python
# Wrong
frequency_days=0

# Correct
frequency_days=7
```

### Very slow execution

Reduce the number of queries:
```python
# Slow (12 dates × 20 samples = 240 calls)
frequency_days=30
n_samples=20

# Faster (4 dates × 10 samples = 40 calls)
frequency_days=90
n_samples=10
```

### High variance in results

Increase n_samples for more stable estimates:
```python
n_samples=20  # Instead of 10
```

Or decrease reword_temperature for less variation:
```python
reword_temperature=0.3  # Instead of 0.5
```

## Best Practices

1. **Start with small tests**: Use `n_samples=3` and short date ranges to test quickly

2. **Choose appropriate frequency**:
   - Short-term questions (days/weeks): `frequency_days=1` or `7`
   - Medium-term (months): `frequency_days=7` or `14`
   - Long-term (years): `frequency_days=30` or `90`

3. **Use meaningful cutoff dates**: Align dates with known events or milestones

4. **Consider API costs**: Each time point costs `n_samples × 2` API calls

5. **Save results**: Cache results to avoid re-running expensive queries

6. **Analyze trends**: Look for gradual changes vs sudden jumps (information shocks)

## References

- Main documentation: `CLAUDE.md`
- Probability estimator: `probability_estimator.py`
- Example scripts: `probability_evolution_example.py`, `test_probability_evolution.py`
- Related: `notebooks/probability_distribution_analysis.ipynb`
