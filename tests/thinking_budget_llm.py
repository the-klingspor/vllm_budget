from unittest.mock import Mock, patch

from vllm_budget.thinking_budget_llm import ThinkingBudgetLLM


class TestThinkingBudgetLLM:
    """Tests for main ThinkingBudgetLLM wrapper."""
    
    @patch('vllm_budget.thinking_budget_llm.LLM')  # Adjust import path as needed
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
        mock_engine = Mock()
        mock_engine.generate.return_value = [Mock(outputs=[Mock(token_ids=[1, 2, 999])])]
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
        
        original = Mock(max_tokens=100, temperature=0.8)
        copy = llm._copy_sampling_params(original)
        
        assert copy is not original
        assert copy.max_tokens == 100