"""
vllm_budget - Thinking budget wrapper for vLLM

A library that extends vLLM with thinking budget support, enabling two-stage
generation where the thinking phase has a token budget and automatically
continues generation if the budget is exhausted.
"""

__version__ = "0.1.0"

from vllm_budget.thinking_budget_llm import ThinkingBudgetLLM
from vllm_budget.config import ThinkingBudgetConfig
from vllm_budget.utils import get_default_early_stopping_text, get_default_think_end_token

__all__ = [
    'ThinkingBudgetLLM',
    'ThinkingBudgetConfig',
    'get_default_early_stopping_text',
    'get_default_think_end_token',
]