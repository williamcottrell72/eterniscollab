# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**eterniscollab** is a forecasting and LLM analysis toolkit. The project provides tools for:
- Generating probability distributions from LLM responses
- Analyzing "buzz" scores (interest × divisiveness) for topics
- Generating forecasting questions from trending news
- Comparing different LLM providers (OpenAI, Claude) via OpenRouter API

## Project Structure

```
eterniscollab/
├── buzz.py                          # Buzz score analysis (interest + divisiveness)
├── probability_estimator.py         # Probability distribution estimation via LLMs
├── generate_topic_rankings.py       # Generate and rank topics by interest/buzz
├── topic_generator.py               # Generate trending topics and forecasting questions
├── utils.py                         # Shared utility functions for LLM API calls
├── requirements.txt                 # Python dependencies
├── notebooks/                       # Jupyter notebooks for analysis
│   ├── probability_distribution_analysis.ipynb
│   └── buzz_distribution_analysis.ipynb
├── tests/                           # Unit tests
│   └── test_topic_generator.py
├── TOPIC_GENERATOR_README.md        # Documentation for topic generator
└── topic_generator_example.py      # Example usage scripts
```

## Core Modules

### 1. **probability_estimator.py**
- **Purpose**: Estimate probability distributions by querying LLMs multiple times with reworded prompts
- **Key Functions**:
  - `get_probability_distribution()`: Main async function for collecting probability samples
  - `reword_prompt()`: Reword prompts with variable flexibility (temperature-controlled)
  - `query_probability()`: Get single probability estimate
- **Technology**: Uses Pydantic AI with OpenRouter API
- **Models Supported**: Any OpenRouter model (openai/gpt-4o-mini, anthropic/claude-sonnet-4, etc.)

### 2. **buzz.py**
- **Purpose**: Calculate "buzz scores" for topics based on interest and divisiveness
- **Key Functions**:
  - `get_buzz_score()`: Calculate combined buzz score (interest × divisiveness)
  - `get_buzz_score_openrouter()`: OpenRouter version
  - `query_interest()`, `query_interest_openrouter()`: Get interest scores
  - `query_divisiveness()`, `query_divisiveness_openrouter()`: Get divisiveness scores
- **Formula**: Buzz = Interest × Divisiveness (both 0-1)
- **Use Case**: Identify topics that are both interesting AND divisive

### 3. **topic_generator.py**
- **Purpose**: Generate trending topics from recent news and create forecasting questions
- **Key Function**: `generate_topics_and_questions(n_topics, k_questions)`
- **Features**:
  - Uses OpenRouter's `:online` feature for real-time web search
  - Generates yes/no questions with specific future resolution dates
  - Enforces topic diversity (no duplicate subject areas)
  - Validates questions are objective and verifiable
- **Default Model**: `openai/gpt-4o-mini:online` (web search enabled)

### 4. **generate_topic_rankings.py**
- **Purpose**: Generate large lists of topics and rank them by interest/buzz
- **Key Functions**:
  - `generate_topics_with_llm()`: Generate N topics using LLM
  - `generate_topics_with_llm_openrouter()`: OpenRouter version
  - `rank_topics_by_interest()`: Rank topics
- **Use Case**: Create ranked lists of 100s-1000s of topics

### 5. **utils.py**
- **Purpose**: Shared utility functions for LLM API calls
- **Key Functions**:
  - `query_llm_for_numeric_value()`: Query LLM and extract numeric value (0-1)
  - `query_llm_for_numeric_value_openrouter()`: OpenRouter version
  - `query_llm_for_text()`: Get raw text responses
  - `extract_numeric_value()`: Parse probabilities, percentages, fractions from text
- **Providers**: OpenAI, Anthropic (Claude), OpenRouter (unified API)

## Jupyter Notebooks

### probability_distribution_analysis.ipynb
- Analyzes probability distributions from LLM responses
- Compares OpenAI vs Claude at different temperatures
- Tests reword_temperature (prompt variation) vs prompt_temperature (response randomness)
- Visualizations: Histograms, comparisons, statistical summaries
- **Location**: `notebooks/probability_distribution_analysis.ipynb`

### buzz_distribution_analysis.ipynb
- Analyzes interest and divisiveness score distributions
- Compares OpenAI vs Claude at different temperatures
- Scatter plots of interest vs divisiveness
- Box plots for statistical comparison
- **Location**: `notebooks/buzz_distribution_analysis.ipynb`

## Key Technologies

- **Python 3.12+**
- **Pydantic AI**: For structured LLM interactions with OpenRouter
- **OpenRouter API**: Unified API for accessing multiple LLM providers
- **Plotly**: Interactive visualizations in notebooks
- **Pytest**: Unit testing framework
- **Jupyter**: For analysis notebooks

## Environment Setup

### Required Environment Variables
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

Optional (for legacy direct API access):
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_topic_generator.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Development Commands

### Run Topic Generator
```bash
# Generate 5 topics with up to 3 questions each (default)
python topic_generator.py

# Generate 2 topics with 1 question each
python topic_generator.py 2 1

# Run example scripts
python topic_generator_example.py
```

### Run Generate Topic Rankings
```bash
# Generate and rank topics
python generate_topic_rankings.py
```

### Launch Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## Architecture & Design Patterns

### OpenRouter Integration
- All modules now support OpenRouter API for unified access to multiple LLM providers
- Model format: `"provider/model-name"` (e.g., `"openai/gpt-4o-mini"`, `"anthropic/claude-sonnet-4"`)
- Web search: Append `:online` to any model (e.g., `"openai/gpt-4o-mini:online"`)
- Cost: $4 per 1,000 web results for `:online` models

### Async/Await Pattern
- probability_estimator.py uses async/await for efficient concurrent LLM queries
- Pydantic AI agents run asynchronously
- Use `await` when calling probability distribution functions

### Temperature Control
Two types of temperature:
1. **Reword Temperature**: Controls prompt variation (0 = no rewording, 1 = high variation)
2. **Prompt Temperature**: Controls LLM response randomness (standard LLM parameter)

### Error Handling
- All modules validate API keys before making requests
- JSON parsing includes fallback regex extraction
- Numeric value extraction handles multiple formats (decimals, percentages, fractions)

## Conventions

### Code Style
- Type hints for all function parameters and return values
- Docstrings following Google style
- Descriptive variable names
- Constants in UPPER_CASE

### Code Organization
  - Group code into directories based on functionality with the goal of minimizing complexity.
  - Attempt to factor out useful functions and code into modules with minimal dependencies cross dependencies.
  - Re-use utility functions such as emas or forward price return construction as much as possible.  Any time you are creating a new utility first check
    to see if it already exists and use the existing function or modify it slightly for the new use-case if appropriate, but in a backwards compatible manner.
  - If a completely new function is needed for either analysis purposes or feature construction, place it in a central place with other utilities for broad re-use.
  - Avoid duplicate code.  If the same code pattern appears in multiple places, see if we can reduce code by centralizing the function.
  - Avoid monolithic classes that try and do too many things at once.  It's better to split out code for fitting, feature building and backtesting so each can be
    addressed and improved separately.  Any classes that combine these functionalities should be calling well-specified external modules handling fitting, feature building or backtesting.



### Testing
- Unit tests for all public functions
- Mock external API calls in tests
- Integration tests marked with `@pytest.mark.skipif` for missing API keys
- Test file naming: `test_<module_name>.py`
- When making large changes to key functions rerun any tests that might be affected.
- When adding a large amount of code or new functions that are used in many places consider adding a test.
- Test must be quick to run.


### Notebooks
- Clear markdown sections explaining each step
- Configuration at the top
- Statistical summaries included
- Visualizations with large, readable fonts (20px titles, 16px labels)
- Figure sizes: 1000-1400px width, 800-1000px height
- **Notebooks should be runnable from start to finish**
- Execute "Run All" should work without errors on a fresh kernel
- Each notebook should be self-contained and complete

## Integration Points

### External APIs
- **OpenRouter**: Primary API gateway for LLM access
- **Exa.ai**: Backend for OpenRouter's web search (via `:online` models)

### Data Flow
```
User Input → Topic Generator → OpenRouter API (with :online) → Web Search
                ↓
           Trending Topics + Questions
                ↓
         Buzz Score Analysis → Interest + Divisiveness
                ↓
       Probability Estimation → Distribution Analysis
```

## Important Notes

### Topic Generator Requirements
- Questions must be **yes/no format**
- Questions must have **future resolution dates** (never in the past)
- Topics must be **diverse** (different domains/subject areas)
- Examples provided in prompts to prevent duplicate topics (e.g., "EU AI Act" and "AI Governance" should be ONE topic)

### Probability Estimation
- First sample always uses original prompt (when reword_temperature > 0)
- Remaining samples use reworded versions
- Results include both probabilities and reworded prompts for analysis

### Buzz Scores
- Interest: How interesting is the topic? (0-1)
- Divisiveness: How divisive/controversial is the topic? (0-1)
- Buzz: Interest × Divisiveness (high buzz = interesting AND divisive)

## Troubleshooting

### Common Issues

1. **"OPENROUTER_API_KEY not set"**
   - Set environment variable before running scripts
   - Check spelling: `OPENROUTER_API_KEY` (not `OPEN_ROUTER_API_KEY`)

2. **"No JSON object found in response"**
   - LLM didn't return valid JSON
   - Check model supports JSON output
   - Increase max_tokens if response is truncated

3. **Plots too small in notebooks**
   - All plotting cells have been updated with larger sizes
   - If issues persist, adjust `height` and `width` parameters in plotting code

4. **Duplicate topics generated**
   - Topic diversity is enforced in prompts
   - If still occurring, model may need stronger instructions or temperature adjustment

## Future Enhancements

Potential areas for expansion:
- Add more providers (Gemini, Llama, etc.) via OpenRouter
- Implement caching for expensive API calls
- Add time-series analysis of how buzz/probabilities change over time
- Create web interface for topic generation
- Add question resolution tracking system

## Contributing

When adding new features:
1. Add type hints and docstrings
2. Write unit tests
3. Update this CLAUDE.md file
4. Add examples to relevant README files
5. Ensure notebooks remain readable (large fonts, clear visualizations)
