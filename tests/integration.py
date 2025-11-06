from unittest.mock import Mock, patch

from vllm_budget.thinking_budget_llm import ThinkingBudgetLLM

class TestIntegration:
    """Integration tests for complete workflows."""
    
    @patch('wrapper.LLM')
    def test_complete_two_stage_flow(self, mock_llm_class):
        """Test complete two-stage generation flow."""
        # Setup mock engine
        mock_engine = Mock()
        
        # First stage: thinking not complete
        first_output = Mock()
        first_output.outputs = [Mock(token_ids=[1, 2, 3, 4, 5])]
        
        # Second stage: completion
        second_output = Mock()
        second_output.outputs = [Mock(token_ids=[10, 20, 1, 2, 3, 4, 5, 6, 7, 2])]
        
        mock_engine.generate.side_effect = [first_output, second_output]
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
