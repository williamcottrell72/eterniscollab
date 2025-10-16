# OpenRouter Integration Setup Guide

## Installation

The required packages have been added to `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pydantic-ai-slim>=0.0.14
pip install pydantic>=2.0.0
pip install httpx>=0.24.0
pip install openai>=1.0.0
```

## Environment Variable Setup

Set your OpenRouter API key as an environment variable:

```bash
# Temporary (current session only)
export OPENROUTER_API_KEY='sk-or-v1-...'

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.bashrc
source ~/.bashrc
```

## Available Models

The following models are pre-configured in `OPENROUTER_MODELS`:

### OpenAI Models
- `openai/gpt-4o` - Latest GPT-4o
- `openai/gpt-4o-mini` - Efficient GPT-4o
- `openai/o1` - Reasoning model
- `openai/o1-mini` - Smaller reasoning model

### Anthropic Claude Models
- `anthropic/claude-sonnet-4` - Latest Claude Sonnet
- `anthropic/claude-3.7-sonnet` - Claude 3.7
- `anthropic/claude-3.5-sonnet` - Claude 3.5

### Qwen Models
- `qwen/qwen-2.5-72b-instruct` - Qwen 2.5 72B
- `qwen/qwq-32b-preview` - Qwen QwQ 32B

## Usage

### Basic Example

```python
import asyncio
from probability_estimator import get_probability_distribution

async def main():
    result = await get_probability_distribution(
        prompt="What is the probability that Kamala Harris runs for president again?",
        model="anthropic/claude-sonnet-4",  # OpenRouter format
        n_samples=10,
        reword_temperature=0.5,
        prompt_temperature=0.7,
        knowledge_cutoff_date="January 2024"  # Optional
    )

    # Access results
    print(f"Probabilities: {result['probabilities']}")
    print(f"Model used: {result['model']}")
    print(f"Reworded prompts: {result['reworded_prompts']}")

# Run async function
asyncio.run(main())
```

### Running the Example Script

```bash
# Make sure environment variable is set
export OPENROUTER_API_KEY='your-api-key-here'

# Run the example
python example_openrouter.py
```

The example script will:
1. Test 3 different models (GPT-4o-mini, Claude 3.5, Qwen 2.5)
2. Run 5 samples per model
3. Display probabilities, statistics, and reworded prompts

### Trying Different Models

```python
# Try GPT-4o
result = await get_probability_distribution(
    "Will it rain tomorrow?",
    model="openai/gpt-4o",
    n_samples=20
)

# Try Claude Sonnet 4
result = await get_probability_distribution(
    "Will it rain tomorrow?",
    model="anthropic/claude-sonnet-4",
    n_samples=20
)

# Try Qwen
result = await get_probability_distribution(
    "Will it rain tomorrow?",
    model="qwen/qwen-2.5-72b-instruct",
    n_samples=20
)
```

## Key Differences from Legacy Function

| Feature | New (OpenRouter) | Legacy |
|---------|------------------|--------|
| Function name | `get_probability_distribution()` | `get_probability_distribution_legacy()` |
| Async/Sync | Async (use `await`) | Async (use `await`) |
| Model format | `"provider/model-name"` | `"provider/model-name"` |
| API used | OpenRouter | OpenRouter |
| Environment var | `OPENROUTER_API_KEY` | `OPENROUTER_API_KEY` |
| Model count | 400+ models | 400+ models |
| Status | Recommended | Deprecated - use `get_probability_distribution()` instead |

## Return Value

Both functions return a dictionary with:

```python
{
    'probabilities': [0.25, 0.30, 0.28, ...],  # List of probability estimates
    'reworded_prompts': ["Prompt 1", "Prompt 2", ...],  # Reworded versions
    'model': 'anthropic/claude-sonnet-4',  # Model used
    'n_samples': 10,  # Number of samples
    'reword_temperature': 0.5,  # Reword temperature
    'prompt_temperature': 0.7,  # Sampling temperature
    'knowledge_cutoff_date': 'January 2024'  # Knowledge cutoff (if provided)
}
```

## Troubleshooting

### Import Error: pydantic-ai not installed

```bash
pip install pydantic-ai-slim openai
```

### Environment Variable Not Set

```bash
echo $OPENROUTER_API_KEY  # Check if set
export OPENROUTER_API_KEY='your-key-here'  # Set it
```

### Async Function Errors

Make sure you're using `await` and `asyncio.run()`:

```python
# ❌ Wrong
result = get_probability_distribution(...)

# ✅ Correct
result = await get_probability_distribution(...)

# ✅ Correct (in script)
asyncio.run(get_probability_distribution(...))
```

## API Key Management

Get your OpenRouter API key:
1. Go to https://openrouter.ai/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new key
5. Set it as environment variable

## Cost Considerations

OpenRouter charges per model usage. Check pricing at:
https://openrouter.ai/models

Different models have different costs:
- `gpt-4o-mini`: Lower cost
- `claude-sonnet-4`: Medium cost
- `gpt-4o`: Higher cost
- `qwen` models: Varies

Choose models based on your budget and performance needs.
