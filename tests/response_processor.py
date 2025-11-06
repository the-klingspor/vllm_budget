from unittest.mock import Mock



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
        # Mock output with think end token
        mock_output = Mock()
        mock_output.outputs = [Mock(token_ids=[1, 2, 999, 5])]  # 999 is think end
        outputs = [mock_output]
        prompts = [[10, 20]]
        early_stopping_tokens = [99, 100]
        
        final_responses, second_stage_prompts, indices, lengths = \
            response_processor.process_first_stage(outputs, prompts, early_stopping_tokens)
        
        assert len(second_stage_prompts) == 1
        assert final_responses[0] is None  # Placeholder
    
    def test_process_first_stage_budget_exhausted(self, response_processor):
        """Test first stage when thinking budget exhausted."""
        # Mock output without EOS or think end token
        mock_output = Mock()
        mock_output.outputs = [Mock(token_ids=[1, 2, 3, 4])]
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
        final_responses = [Mock(token_ids=[1, 2]), None, None]
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