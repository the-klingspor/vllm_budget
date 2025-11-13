"""
Comprehensive test suite for vllm_thinking_budget library.
Focus on edge cases and critical functionality.
"""

import pytest
from unittest.mock import Mock, patch

# Import modules (adjust based on actual package structure)
from vllm_budget.config import ThinkingBudgetConfig
from vllm_budget.token_detector import TokenDetector
from vllm_budget.response_processor import ResponseProcessor
from vllm_budget.thinking_budget_llm import ThinkingBudgetLLM
from vllm_budget.utils import get_default_early_stopping_text, get_default_think_end_token


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


# ============================================================================
# Configuration Tests
# ============================================================================

class TestThinkingBudgetConfig:
    """Tests for ThinkingBudgetConfig."""
    
    def test_valid_config(self):
        """Test creation of valid configuration."""
        config = ThinkingBudgetConfig(
            thinking_budget=100,
            early_stopping_text="Stop here",
            think_end_token="</think>"
        )
        assert config.thinking_budget == 100
        assert config.early_stopping_text == "Stop here"
    
    def test_zero_thinking_budget(self):
        """Test that zero thinking budget raises error."""
        with pytest.raises(ValueError):
            config = ThinkingBudgetConfig(
                thinking_budget=0,
                early_stopping_text="Stop"
            )
            config.validate()
    
    def test_negative_thinking_budget(self):
        """Test that negative thinking budget raises error."""
        with pytest.raises(ValueError):
            config = ThinkingBudgetConfig(
                thinking_budget=-10,
                early_stopping_text="Stop"
            )
            config.validate()
    
    def test_empty_early_stopping_text(self):
        """Test that empty early stopping text raises error."""
        with pytest.raises(ValueError):
            config = ThinkingBudgetConfig(
                thinking_budget=100,
                early_stopping_text=""
            )
            config.validate()
    
    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "thinking_budget": 100,
            "early_stopping_text": "Stop",
            "think_end_token": "</think>"
        }
        config = ThinkingBudgetConfig.from_dict(config_dict)
        assert config.thinking_budget == 100
        assert config.think_end_token == "</think>"
    
    def test_missing_required_fields(self):
        """Test that missing required fields raises error."""
        with pytest.raises((TypeError, KeyError)):
            ThinkingBudgetConfig.from_dict({"thinking_budget": 100})


# ============================================================================
# TokenDetector Tests
# ============================================================================

class TestTokenDetector:
    """Tests for TokenDetector."""
    
    def test_get_eos_token_id(self, token_detector):
        """Test retrieval of EOS token ID."""
        eos_id = token_detector.get_eos_token_id()
        assert eos_id == 2
    
    def test_get_think_end_token_id(self, token_detector):
        """Test retrieval of think end token ID."""
        think_id = token_detector.get_think_end_token_id()
        assert think_id == 999
    
    def test_has_eos_token_present(self, token_detector):
        """Test detection when EOS token is present."""
        token_ids = [1, 5, 10, 2, 15]
        assert token_detector.has_eos_token(token_ids) is True
    
    def test_has_eos_token_absent(self, token_detector):
        """Test detection when EOS token is absent."""
        token_ids = [1, 5, 10, 15]
        assert token_detector.has_eos_token(token_ids) is False
    
    def test_has_think_end_token_present(self, token_detector):
        """Test detection when think end token is present."""
        token_ids = [1, 5, 999, 15]
        assert token_detector.has_think_end_token(token_ids) is True
    
    def test_has_think_end_token_absent(self, token_detector):
        """Test detection when think end token is absent."""
        token_ids = [1, 5, 10, 15]
        assert token_detector.has_think_end_token(token_ids) is False
    
    def test_encode_early_stopping_text(self, token_detector):
        """Test encoding of early stopping text."""
        text = "Stop thinking"
        encoded = token_detector.encode_early_stopping_text(text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0
    
    def test_empty_token_list(self, token_detector):
        """Test token detection with empty list."""
        assert token_detector.has_eos_token([]) is False
        assert token_detector.has_think_end_token([]) is False
    
    def test_single_token_eos(self, token_detector):
        """Test with single EOS token."""
        assert token_detector.has_eos_token([2]) is True
    
    def test_no_think_token_configured(self, mock_tokenizer):
        """Test detector without think end token configured."""
        detector = TokenDetector(tokenizer=mock_tokenizer)
        assert detector.get_think_end_token_id() is None
        assert detector.has_think_end_token([1, 2, 3]) is False


# ============================================================================
# ResponseProcessor Tests
# ============================================================================

class TestResponseProcessor:
    """Tests for ResponseProcessor."""
    
    def test_get_prompt_token_length_string(self, response_processor, mock_tokenizer):
        """Test token length calculation for string prompt."""
        prompt = "This is a test"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        length = response_processor.get_prompt_token_length(prompt)
        assert length == 4
    
    def test_get_prompt_token_length_tokens(self, response_processor):
        """Test token length calculation for token list prompt."""
        prompt = [1, 2, 3, 4, 5]
        length = response_processor.get_prompt_token_length(prompt)
        assert length == 5
    
    def test_reconstruct_prompt_string_format(self, response_processor, mock_tokenizer):
        """Test prompt reconstruction with string input."""
        prompt = "Original prompt"
        first_tokens = [10, 20, 30]
        mock_tokenizer.encode.return_value = [1, 2]
        mock_tokenizer.decode.return_value = "Reconstructed"
        
        result = response_processor.reconstruct_prompt(prompt, first_tokens)
        assert isinstance(result, str)
    
    def test_reconstruct_prompt_token_format(self, response_processor):
        """Test prompt reconstruction with token input."""
        prompt = [1, 2, 3]
        first_tokens = [10, 20, 30]
        
        result = response_processor.reconstruct_prompt(prompt, first_tokens)
        assert isinstance(result, list)
        assert result == [1, 2, 3, 10, 20, 30]
    
    def test_reconstruct_prompt_with_additional_tokens(self, response_processor):
        """Test prompt reconstruction with additional tokens."""
        prompt = [1, 2, 3]
        first_tokens = [10, 20]
        additional_tokens = [30, 40]
        
        result = response_processor.reconstruct_prompt(
            prompt, first_tokens, additional_tokens
        )
        assert result == [1, 2, 3, 10, 20, 30, 40]
    
    def test_process_first_stage_all_complete(self, response_processor):
        """Test first stage processing when all outputs complete."""
        # Mock outputs with EOS tokens
        mock_output = Mock()
        mock_output.outputs = [Mock(token_ids=[1, 2, 3, 2])]  # 2 is EOS
        outputs = [mock_output]
        prompts = [[10, 20]]
        early_stopping_tokens = [99, 100]
        
        final_responses, second_stage_prompts, indices, lengths = \
            response_processor.process_first_stage(outputs, prompts, early_stopping_tokens)
        
        assert len(final_responses) == 1
        assert len(second_stage_prompts) == 0
        assert final_responses[0] == [1, 2, 3, 2]
    
    def test_process_first_stage_thinking_complete(self, response_processor):
        """Test first stage when thinking completes naturally."""
        # Mock output with think end token (avoiding EOS token id=2)
        mock_output = Mock()
        mock_output.outputs = [Mock(token_ids=[1, 3, 999, 5])]  # 999 is think end, no EOS (2)
        outputs = [mock_output]
        prompts = [[10, 20]]
        early_stopping_tokens = [99, 100]
        
        final_responses, second_stage_prompts, indices, lengths = \
            response_processor.process_first_stage(outputs, prompts, early_stopping_tokens)

        assert final_responses[0] is None  # Placeholder
        assert len(second_stage_prompts) == 1, "Thinking finished, but no EOS found"
    
    def test_process_first_stage_budget_exhausted(self, response_processor):
        """Test first stage when thinking budget exhausted."""
        # Mock output without EOS or think end token
        mock_output = Mock()
        mock_output.outputs = [Mock(token_ids=[3, 4, 5, 6])]  # No EOS (2) or think end (999)
        outputs = [mock_output]
        prompts = [[10, 20]]
        early_stopping_tokens = [99, 100]
        
        final_responses, second_stage_prompts, indices, lengths = \
            response_processor.process_first_stage(outputs, prompts, early_stopping_tokens)
        
        assert len(second_stage_prompts) == 1
        assert final_responses[0] is None
    
    def test_process_first_stage_mixed_batch(self, response_processor):
        """Test first stage with mixed completion states in batch."""
        # First completes, second needs continuation
        mock_output1 = Mock()
        mock_output1.outputs = [Mock(token_ids=[1, 2, 2])]  # Complete
        mock_output2 = Mock()
        mock_output2.outputs = [Mock(token_ids=[3, 4, 5])]  # Needs continuation
        
        outputs = [mock_output1, mock_output2]
        prompts = [[10, 20], [30, 40]]
        early_stopping_tokens = [99]
        
        final_responses, second_stage_prompts, indices, lengths = \
            response_processor.process_first_stage(outputs, prompts, early_stopping_tokens)
        
        assert len(final_responses) == 2
        assert final_responses[0] == [1, 2, 2]
        assert final_responses[1] is None
        assert len(second_stage_prompts) == 1
    
    def test_process_first_stage_multiple_samples_per_prompt(self, response_processor):
        """Test first stage with multiple samples (n>1)."""
        # One prompt, two samples
        mock_output = Mock()
        mock_output.outputs = [
            Mock(token_ids=[1, 2, 2]),  # Complete
            Mock(token_ids=[3, 4, 5])   # Needs continuation
        ]
        outputs = [mock_output]
        prompts = [[10, 20]]
        early_stopping_tokens = [99]
        
        final_responses, second_stage_prompts, indices, lengths = \
            response_processor.process_first_stage(outputs, prompts, early_stopping_tokens)
        
        assert len(final_responses) == 2
        assert final_responses[0] == [1, 2, 2]
        assert final_responses[1] is None
    
    def test_process_second_stage(self, response_processor):
        """Test second stage processing."""
        # Mock second stage output
        mock_output = Mock()
        mock_output.outputs = [Mock(token_ids=[10, 20, 30, 40, 50])]  # Includes original prompt
        outputs = [mock_output]
        
        final_responses = [None]
        second_stage_indices = [0]
        original_prompt_lengths = [2]  # First 2 tokens are prompt
        
        result = response_processor.process_second_stage(
            outputs, final_responses, second_stage_indices, original_prompt_lengths
        )
        
        assert result[0] == [30, 40, 50]  # Only new tokens after prompt
    
    def test_process_second_stage_multiple_outputs(self, response_processor):
        """Test second stage with multiple incomplete outputs."""
        mock_output1 = Mock()
        mock_output1.outputs = [Mock(token_ids=[10, 20, 30, 40])]
        mock_output2 = Mock()
        mock_output2.outputs = [Mock(token_ids=[50, 60, 70, 80])]
        
        outputs = [mock_output1, mock_output2]
        final_responses = [[1, 2], None, None]  # First response complete from first stage
        second_stage_indices = [1, 2]
        original_prompt_lengths = [1, 2]
        
        result = response_processor.process_second_stage(
            outputs, final_responses, second_stage_indices, original_prompt_lengths
        )
        
        assert result[0] == [1, 2]  # Unchanged from first stage
        assert result[1] == [20, 30, 40]
        assert result[2] == [70, 80]
    
    def test_process_second_stage_empty_new_tokens(self, response_processor):
        """Test second stage when no new tokens generated."""
        mock_output = Mock()
        mock_output.outputs = [Mock(token_ids=[10, 20])]  # Only prompt tokens
        outputs = [mock_output]
        
        final_responses = [None]
        second_stage_indices = [0]
        original_prompt_lengths = [2]
        
        result = response_processor.process_second_stage(
            outputs, final_responses, second_stage_indices, original_prompt_lengths
        )
        
        assert result[0] == []


# ============================================================================
# ThinkingBudgetLLM Tests
# ============================================================================

class TestThinkingBudgetLLM:
    """Tests for main ThinkingBudgetLLM wrapper."""
    
    @patch('vllm_budget.thinking_budget_llm.LLM')
    def test_initialization(self, mock_llm_class):
        """Test basic initialization."""
        llm = ThinkingBudgetLLM(
            model="test-model",
            thinking_budget=100,
            early_stopping_text="Stop"
        )
        assert llm is not None
    
    @patch('vllm_budget.thinking_budget_llm.LLM')
    def test_from_vllm(self, mock_llm_class):
        """Test creation from existing vLLM instance."""
        mock_vllm = Mock()
        llm = ThinkingBudgetLLM.from_vllm(
            mock_vllm,
            thinking_budget=100
        )
        assert llm is not None
    
    def test_normalize_prompts_single_string(self):
        """Test prompt normalization with single string."""
        llm = Mock(spec=ThinkingBudgetLLM)
        llm._normalize_prompts = ThinkingBudgetLLM._normalize_prompts.__get__(llm)
        
        result = llm._normalize_prompts("Single prompt")
        assert result == ["Single prompt"]
    
    def test_normalize_prompts_list_strings(self):
        """Test prompt normalization with list of strings."""
        llm = Mock(spec=ThinkingBudgetLLM)
        llm._normalize_prompts = ThinkingBudgetLLM._normalize_prompts.__get__(llm)
        
        prompts = ["Prompt 1", "Prompt 2"]
        result = llm._normalize_prompts(prompts)
        assert result == prompts
    
    def test_normalize_prompts_token_ids(self):
        """Test prompt normalization with token IDs."""
        llm = Mock(spec=ThinkingBudgetLLM)
        llm._normalize_prompts = ThinkingBudgetLLM._normalize_prompts.__get__(llm)
        
        prompts = [[1, 2, 3], [4, 5, 6]]
        result = llm._normalize_prompts(prompts)
        assert result == prompts
    
    @patch('vllm_budget.thinking_budget_llm.LLM')
    def test_generate_without_thinking_budget(self, mock_llm_class):
        """Test standard generation when no thinking budget specified."""
        mock_engine = Mock()
        mock_llm_class.return_value = mock_engine
        
        llm = ThinkingBudgetLLM(model="test-model")
        sampling_params = Mock()
        
        llm.generate(prompts=["Test"], sampling_params=sampling_params)
        # Should call standard vLLM generate
        mock_engine.generate.assert_called_once()
    
    @patch('vllm_budget.thinking_budget_llm.LLM')
    def test_generate_with_thinking_budget(self, mock_llm_class):
        """Test generation with thinking budget."""
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens=False: 
                                     [100 + i for i in range(len(text.split()))])
        mock_tokenizer.decode = Mock(side_effect=lambda ids, skip_special_tokens=False: 
                                     " ".join([f"word{i}" for i in ids]))
        
        # Create mock engine
        mock_engine = Mock()
        mock_engine.get_tokenizer = Mock(return_value=mock_tokenizer)
        mock_engine.generate.return_value = [Mock(outputs=[Mock(token_ids=[1, 3, 999])])]  # No EOS
        mock_llm_class.return_value = mock_engine
        
        llm = ThinkingBudgetLLM(
            model="test-model",
            thinking_budget=100,
            think_end_token_id=999
        )
        sampling_params = Mock()
        sampling_params.max_tokens = 200
        
        result = llm.generate(
            prompts=["Test"],
            sampling_params=sampling_params,
            thinking_budget=50
        )
        
        # Should call generate at least once (first stage)
        assert mock_engine.generate.call_count >= 1
    
    def test_copy_sampling_params(self):
        """Test copying of sampling parameters."""
        llm = Mock(spec=ThinkingBudgetLLM)
        llm._copy_sampling_params = ThinkingBudgetLLM._copy_sampling_params.__get__(llm)
        
        # Create mock without copy method to test deepcopy path
        original = Mock(spec=['max_tokens', 'temperature'])
        original.max_tokens = 100
        original.temperature = 0.8
        copy = llm._copy_sampling_params(original)
        
        assert copy is not original
        assert copy.max_tokens == 100
        assert copy.temperature == 0.8


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    @patch('vllm_budget.thinking_budget_llm.LLM')
    def test_complete_two_stage_flow(self, mock_llm_class):
        """Test complete two-stage generation flow."""
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens=False: 
                                     [100 + i for i in range(len(text.split()))])
        mock_tokenizer.decode = Mock(side_effect=lambda ids, skip_special_tokens=False: 
                                     " ".join([f"word{i}" for i in ids]))
        
        # Setup mock engine
        mock_engine = Mock()
        mock_engine.get_tokenizer = Mock(return_value=mock_tokenizer)
        
        # First stage: thinking not complete (no EOS, no think end)
        first_output = Mock()
        first_output.outputs = [Mock(token_ids=[1, 3, 4, 5, 6])]  # Avoid EOS (2)
        
        # Second stage: completion
        second_output = Mock()
        second_output.outputs = [Mock(token_ids=[10, 20, 1, 3, 4, 5, 6, 7, 8, 2])]  # With EOS
        
        mock_engine.generate.side_effect = [[first_output], [second_output]]
        mock_llm_class.return_value = mock_engine
        
        llm = ThinkingBudgetLLM(
            model="test-model",
            thinking_budget=5,
            early_stopping_text="</think>",
            think_end_token_id=999
        )
        
        sampling_params = Mock(max_tokens=20)
        result = llm.generate(
            prompts=["Test prompt"],
            sampling_params=sampling_params,
            thinking_budget=5
        )
        
        # Should have called generate twice
        assert mock_engine.generate.call_count == 2
    
    def test_edge_case_thinking_budget_equals_max_tokens(self):
        """Test when thinking budget equals max tokens."""
        # This should handle gracefully - no second stage possible
        pass
    
    def test_edge_case_very_small_thinking_budget(self):
        """Test with thinking budget smaller than early stopping text."""
        # Should still work but may hit budget immediately
        pass
    
    def test_edge_case_empty_first_stage_output(self):
        """Test when first stage produces no tokens."""
        pass


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_default_early_stopping_text(self):
        """Test default early stopping text retrieval."""
        text = get_default_early_stopping_text()
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_get_default_think_end_token(self):
        """Test default think end token retrieval."""
        token = get_default_think_end_token()
        assert isinstance(token, str)
        assert len(token) > 0