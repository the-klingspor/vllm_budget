# vllm_budget

A Python library that extends vLLM with thinking budget support, enabling two-stage generation where the thinking phase has a token budget and automatically continues generation if the budget is exhausted.

## Features

- **Two-stage generation**: Allocate a specific token budget for the "thinking" phase
- **Automatic continuation**: Seamlessly continues generation when thinking budget is exhausted
- **Flexible configuration**: Customize early stopping text, think end tokens, and budgets
- **Drop-in replacement**: Mirrors vLLM's API for easy adoption
- **Batch support**: Handles batched prompts and multiple samples per prompt

## Installation

```bash
pip install vllm_budget
```

Or install from source:

```bash
git clone https://github.com/yourusername/vllm_budget.git
cd vllm_budget
pip install -e .
```

## Quick Start

### Basic Usage

```python
from vllm_budget import ThinkingBudgetLLM
from vllm import SamplingParams

# Initialize with thinking budget
llm = ThinkingBudgetLLM(
    model="deepseek-ai/DeepSeek-R1",
    thinking_budget=1000,  # Allocate 1000 tokens for thinking
)

# Generate with budget constraint
sampling_params = SamplingParams(temperature=0.7, max_tokens=2000)
outputs = llm.generate(
    prompts=["Solve this complex problem..."],
    sampling_params=sampling_params
)

# Access results
for output in outputs:
    print(output.outputs[0].text)
```

### Using Existing vLLM Instance

```python
from vllm import LLM
from vllm_budget import ThinkingBudgetLLM

# Create vLLM instance
base_llm = LLM(model="deepseek-ai/DeepSeek-R1", tensor_parallel_size=2)

# Wrap with thinking budget support
llm = ThinkingBudgetLLM.from_vllm(
    base_llm,
    thinking_budget=1000
)
```

### Custom Configuration

```python
llm = ThinkingBudgetLLM(
    model="deepseek-ai/DeepSeek-R1",
    thinking_budget=500,
    early_stopping_text="\n\nTime's up! Final answer:\n</think>\n\n",
    think_end_token="</think>",
    think_end_token_id=12345,  # Optional: specify token ID directly
)
```

### Per-Call Budget Override

```python
# Override default budget for specific calls
outputs = llm.generate(
    prompts=["Quick question..."],
    sampling_params=sampling_params,
    thinking_budget=100,  # Use smaller budget for this call
)

# Disable thinking budget for specific calls
outputs = llm.generate(
    prompts=["Another question..."],
    sampling_params=sampling_params,
    thinking_budget=None,  # Use standard generation
)
```

## How It Works

1. **Stage 1**: Generate up to the thinking budget
   - If generation completes (EOS token found): Return immediately
   - If think end token found: Continue to stage 2
   - If budget exhausted: Insert early stopping text and continue to stage 2

2. **Stage 2**: Complete the generation
   - Continue from where stage 1 left off
   - Use remaining token budget (max_tokens - thinking_budget)

## API Reference

### ThinkingBudgetLLM

```python
ThinkingBudgetLLM(
    model: str,
    thinking_budget: Optional[int] = None,
    early_stopping_text: Optional[str] = None,
    think_end_token: Optional[str] = None,
    think_end_token_id: Optional[int] = None,
    **vllm_kwargs
)
```

**Parameters:**
- `model`: Model name or path
- `thinking_budget`: Default thinking budget in tokens (optional)
- `early_stopping_text`: Text to insert when budget exhausted (optional)
- `think_end_token`: String representation of think end token (optional)
- `think_end_token_id`: Token ID for think end token (optional)
- `**vllm_kwargs`: Additional arguments passed to vLLM LLM constructor

### generate()

```python
generate(
    prompts: Union[str, List[str], List[List[int]]],
    sampling_params: SamplingParams,
    thinking_budget: Optional[int] = None,
    use_tqdm: bool = True
) -> List[RequestOutput]
```

**Parameters:**
- `prompts`: Input prompts (strings or token IDs)
- `sampling_params`: vLLM SamplingParams instance
- `thinking_budget`: Override default budget for this call (optional)
- `use_tqdm`: Whether to show progress bar

**Returns:**
- List of vLLM RequestOutput objects

## Configuration

### ThinkingBudgetConfig

For advanced configuration, use the `ThinkingBudgetConfig` class:

```python
from vllm_budget import ThinkingBudgetConfig

config = ThinkingBudgetConfig(
    thinking_budget=1000,
    early_stopping_text="</think>",
    think_end_token="</think>",
    think_end_token_id=999
)

# Validate configuration
config.validate()

# Create from dictionary
config = ThinkingBudgetConfig.from_dict({
    "thinking_budget": 1000,
    "early_stopping_text": "</think>"
})
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=vllm_budget --cov-report=html
```

### Project Structure

```
vllm_budget/
├── config.py               # Configuration classes
├── token_detector.py       # Token detection logic
├── response_processor.py   # Multi-stage response handling
├── thinking_budget_llm.py  # Main ThinkingBudgetLLM class
└── utils.py                # Utility functions

tests/
└── test_thinking_budget.py  # Comprehensive test suite
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Citation

If you use this library in your research, please cite:

```bibtex
@software{vllm_budget,
  title = {vllm_budget: Thinking Budget Wrapper for vLLM},
  author = {Joschka Strüber},
  year = {2025},
  url = {https://github.com/yourusername/vllm_budget}
}
```

## Acknowledgments

Built on top of the excellent [vLLM](https://github.com/vllm-project/vllm) library.