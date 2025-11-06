import pytest
from unittest.mock import Mock, MagicMock, patch

# Import modules (adjust based on actual package structure)
from vllm_budget.config import ThinkingBudgetConfig
from vllm_budget.token_detector import TokenDetector
from vllm_budget.response_processor import ResponseProcessor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with basic functionality."""
    tokenizer = Mock()
    tokenizer.eos_token_id = 2
    tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens=False, return_tensors=None: 
                           [100 + i for i in range(len(text.split()))])  # Simple word-based encoding
    tokenizer.decode = Mock(side_effect=lambda ids, skip_special_tokens=False: 
                           " ".join([f"word{i}" for i in ids]))
    return tokenizer


@pytest.fixture
def mock_vllm_engine():
    """Create a mock vLLM engine."""
    engine = Mock()
    engine.generate = Mock()
    return engine


@pytest.fixture
def basic_config():
    """Basic valid configuration."""
    return ThinkingBudgetConfig(
        thinking_budget=100,
        early_stopping_text="\n\n</think>\n\n",
        think_end_token="</think>",
        think_end_token_id=999
    )


@pytest.fixture
def token_detector(mock_tokenizer):
    """Create TokenDetector with mock tokenizer."""
    return TokenDetector(
        tokenizer=mock_tokenizer,
        think_end_token="</think>",
        think_end_token_id=999
    )


@pytest.fixture
def response_processor(mock_tokenizer, token_detector):
    """Create ResponseProcessor with dependencies."""
    return ResponseProcessor(
        tokenizer=mock_tokenizer,
        token_detector=token_detector
    )