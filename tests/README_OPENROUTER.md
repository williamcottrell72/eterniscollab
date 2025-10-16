# OpenRouter Integration Tests

Unit tests for the OpenRouter integration with Pydantic AI for probability estimation.

## Overview

The `test_openrouter.py` file contains comprehensive tests for the async `get_probability_distribution` function that uses OpenRouter to access multiple LLM providers through a unified interface.

## Test Classes

### 1. **TestOpenRouterModels**
Tests basic functionality with different OpenRouter models:
- `test_openai_gpt4o_mini`: Tests with OpenAI GPT-4o-mini
- `test_anthropic_claude_sonnet`: Tests with Anthropic Claude Sonnet
- `test_qwen_model`: Tests with Qwen model
- `test_knowledge_cutoff`: Tests knowledge cutoff date constraint

### 2. **TestOpenRouterStatistics**
Tests statistical properties of probability distributions:
- `test_distribution_statistics`: Validates mean, median, std dev, min, max
- `test_reword_temperature_effect`: Tests prompt diversity with different temperatures

### 3. **TestOpenRouterComparison**
Compares results across different models:
- `test_compare_models`: Side-by-side comparison of OpenAI vs Claude

### 4. **TestOpenRouterErrorHandling**
Tests error handling:
- `test_missing_api_key`: Verifies proper error when API key is missing
- `test_invalid_model`: Tests handling of invalid model names

### 5. **TestOpenRouterModelsConstant**
Tests the `OPENROUTER_MODELS` constant:
- `test_models_list_exists`: Validates list is defined
- `test_models_have_correct_format`: Checks `provider/model-name` format
- `test_models_include_major_providers`: Ensures major providers present

## Requirements

### Environment Variables
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

### Python Packages
```bash
pip install -r requirements.txt
```

Key dependencies:
- `pydantic-ai-slim[openai]>=0.0.14` - For OpenRouter integration
- `pytest-asyncio>=0.21.0` - For async test support
- `pytest>=7.0.0` - Test framework
- `numpy>=1.24.0` - For statistics

## Running Tests

### Run all OpenRouter tests:
```bash
pytest tests/test_openrouter.py -v -s
```

### Run specific test class:
```bash
pytest tests/test_openrouter.py::TestOpenRouterModels -v -s
pytest tests/test_openrouter.py::TestOpenRouterStatistics -v -s
```

### Run specific test:
```bash
pytest tests/test_openrouter.py::TestOpenRouterModels::test_openai_gpt4o_mini -v -s
```

### Run tests that don't require API key:
```bash
pytest tests/test_openrouter.py::TestOpenRouterModelsConstant -v
```

### Skip tests without API key:
Tests automatically skip if `OPENROUTER_API_KEY` is not set.

## Test Output

Tests with `-s` flag show detailed output:
```
TestOpenRouterModels::test_openai_gpt4o_mini

openai/gpt-4o-mini Results:
  Probabilities: ['0.450', '0.420', '0.480']
  Mean: 0.450
```

## What Gets Tested

1. **API Integration**
   - Successful API calls to OpenRouter
   - Proper request formatting
   - Response parsing

2. **Model Support**
   - OpenAI models (gpt-4o-mini, o1)
   - Anthropic models (claude-sonnet-4, claude-3.5-sonnet)
   - Other providers (Qwen)

3. **Functionality**
   - Probability extraction (0-1 range)
   - Prompt rewording with temperature control
   - Multiple samples collection
   - Knowledge cutoff date enforcement

4. **Data Quality**
   - Valid probability values
   - Correct data structure
   - Statistical properties (mean, std, etc.)
   - Prompt diversity with rewording

5. **Error Handling**
   - Missing API key detection
   - Invalid model handling
   - Graceful error messages

## Comparison with Example Script

This test file replaces `example_openrouter.py` with:
- ✅ **Automated testing** instead of manual script
- ✅ **Assertions** to verify correctness
- ✅ **Multiple test scenarios** organized by class
- ✅ **Reusable fixtures** for API key
- ✅ **Better error handling** with pytest
- ✅ **Detailed output** with `-s` flag
- ✅ **Skip logic** when API key not available

## Test Coverage

- **Basic functionality**: 4 tests
- **Statistics**: 2 tests
- **Comparison**: 1 test
- **Error handling**: 2 tests
- **Constants**: 3 tests

**Total**: 12 tests

## Notes

- Tests use small sample sizes (2-5) for speed
- Increase `n_samples` for production use (10-20+)
- Some tests may take several seconds due to API calls
- Tests are async and use `pytest-asyncio`
- All async tests marked with `@pytest.mark.asyncio`

## Example Test Run

```bash
$ pytest tests/test_openrouter.py::TestOpenRouterModels -v -s

tests/test_openrouter.py::TestOpenRouterModels::test_openai_gpt4o_mini PASSED

openai/gpt-4o-mini Results:
  Probabilities: ['0.450', '0.420', '0.480']
  Mean: 0.450

tests/test_openrouter.py::TestOpenRouterModels::test_anthropic_claude_sonnet PASSED

anthropic/claude-3.5-sonnet Results:
  Probabilities: ['0.400', '0.380', '0.420']
  Mean: 0.400

========================== 2 passed in 15.3s ==========================
```

## Integration with CI/CD

For continuous integration, you can:
1. Skip tests without API key (automatic)
2. Set `OPENROUTER_API_KEY` in CI secrets
3. Run full test suite on PR/merge

Example GitHub Actions:
```yaml
- name: Run OpenRouter tests
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  run: pytest tests/test_openrouter.py -v
```
