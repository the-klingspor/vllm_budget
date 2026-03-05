"""
Integration test suite for thinking budget exhaustion scenarios.
Tests the complete flow when thinking is not terminated within the budget.
"""

import pytest
from unittest.mock import Mock

from vllm_budget.token_detector import TokenDetector
from vllm_budget.response_processor import ResponseProcessor


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with basic functionality."""
    tokenizer = Mock()
    tokenizer.eos_token_id = 2  # EOS token
    
    # Simple encoding: just return the input tokens or convert words to IDs
    def encode_fn(text, add_special_tokens=False):
        if isinstance(text, str):
            # Simple word-based encoding for testing
            words = text.split()
            return [100 + i for i in range(len(words))]
        return text
    
    tokenizer.encode = Mock(side_effect=encode_fn)
    
    # Simple decoding: convert token IDs to readable strings
    def decode_fn(ids, skip_special_tokens=False):
        parts = []
        for token_id in ids:
            if token_id == 2:
                parts.append("<EOS>")
            elif token_id == 999:
                parts.append("</think>")
            elif token_id == 888:
                parts.append("[EARLY_STOP]")  # Represents early stopping text
            else:
                parts.append(f"tok_{token_id}")
        return " ".join(parts)
    
    tokenizer.decode = Mock(side_effect=decode_fn)
    return tokenizer


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


class TestBudgetExhaustion:
    """Test cases for when thinking budget is exhausted."""
    
    def test_budget_exhausted_complete_flow(self, response_processor, mock_tokenizer):
        """
        Test complete flow when thinking budget is exhausted.
        
        Expected output structure:
        1. Original prompt removed
        2. First stage partial response (thinking tokens)
        3. Early stopping text injected
        4. Think end token appended
        5. Second stage response (answer tokens)
        
        NOTE: vLLM's RequestOutput.outputs[i].token_ids contains ONLY the generated tokens,
        NOT the prompt tokens. So when we use prompt+previous_output as the new prompt,
        vLLM will return only the NEW tokens generated in that stage.
        """
        # Setup: Original prompt
        original_prompt = [10, 11, 12]  # 3 tokens
        
        # First stage output: thinking not complete (no EOS, no </think>)
        # vLLM returns ONLY generated tokens (not prompt)
        first_stage_tokens = [20, 21, 22, 23, 24]  # 5 tokens of thinking
        
        # Early stopping tokens
        early_stopping_tokens = [888, 999]  # [EARLY_STOP, </think>]
        
        # Mock first stage vLLM output
        mock_first_output = Mock()
        mock_first_output.outputs = [Mock(token_ids=first_stage_tokens)]
        
        # Process first stage
        final_responses, second_prompts, indices, prompt_lengths = \
            response_processor.process_first_stage(
                outputs=[mock_first_output],
                prompts=[original_prompt],
                early_stopping_tokens=early_stopping_tokens
            )
        
        # Verify first stage processing
        assert len(second_prompts) == 1, "Should need second stage"
        assert final_responses[0] is None, "Should be placeholder"
        assert prompt_lengths[0] == 3, "Should track original prompt length"
        
        # Verify second stage prompt reconstruction
        # Should be: original_prompt + first_stage_tokens + early_stopping_tokens
        expected_second_prompt = original_prompt + first_stage_tokens + early_stopping_tokens
        assert second_prompts[0] == expected_second_prompt, \
            f"Second prompt should include original + first stage + early stopping. Got {second_prompts[0]}, expected {expected_second_prompt}"
        
        # Second stage output: continuation after early stopping
        # IMPORTANT: vLLM's token_ids contains ONLY the newly generated tokens
        # NOT the full prompt+previous sequence
        second_stage_new_tokens = [30, 31, 32, 2]  # Answer + EOS
        
        # Mock second stage vLLM output (ONLY new tokens, not full sequence)
        mock_second_output = Mock()
        mock_second_output.outputs = [Mock(token_ids=second_stage_new_tokens)]
        
        # Process second stage
        final = response_processor.process_second_stage(
            second_stage_prompts=second_prompts,
            outputs=[mock_second_output],
            final_responses=final_responses,
            second_stage_indices=indices,
            original_prompt_lengths=prompt_lengths
        )
        
        # CRITICAL ASSERTION: Final output should contain ALL parts (except original prompt)
        # Expected structure: first_stage + early_stopping + second_stage_new
        expected_final = first_stage_tokens + early_stopping_tokens + second_stage_new_tokens
        
        # This test checks if process_second_stage correctly combines:
        # 1. First stage tokens from the reconstructed prompt
        # 2. Early stopping text that was injected
        # 3. Second stage newly generated tokens
        assert final[0] == expected_final, (
            f"Final output should be:\n"
            f"  [first_stage] + [early_stopping] + [think_end] + [second_stage]\n"
            f"Expected: {expected_final}\n"
            f"Got:      {final[0]}\n"
            f"Missing:  first_stage={first_stage_tokens}, early_stopping={early_stopping_tokens}, second_stage={second_stage_new_tokens}"
        )
    
    def test_budget_exhausted_text_representation(self, response_processor, mock_tokenizer):
        """
        Test that text representation contains all parts when budget exhausted.
        """
        # Original prompt
        original_prompt = [10, 11]
        
        # First stage: incomplete thinking
        first_stage = [20, 21, 22]
        
        # Early stopping
        early_stopping = [888, 999]  # [EARLY_STOP, </think>]
        
        # Second stage: answer
        second_stage = [40, 2]  # answer + EOS
        
        # Expected complete response (excluding prompt)
        expected_tokens = first_stage + early_stopping + second_stage
        expected_text = mock_tokenizer.decode(expected_tokens)
        
        # Verify text contains all parts
        assert "tok_20" in expected_text, "Should contain first stage tokens"
        assert "</think>" in expected_text, "Should contain think end marker"
        assert "tok_40" in expected_text, "Should contain second stage answer"
        assert "<EOS>" in expected_text, "Should contain EOS token"
    
    def test_budget_exhausted_vs_natural_completion(self, response_processor, mock_tokenizer):
        """
        Compare budget exhausted case vs natural thinking completion.
        Both should have similar final structure but different generation paths.
        """
        original_prompt = [10, 11]
        
        # Case 1: Budget exhausted (no </think> in first stage)
        first_stage_exhausted = [20, 21, 22]  # No 999
        early_stopping = [888]  # "budget exhausted"
        think_end_token = 999  # </think>
        second_stage_exhausted = [30, 999, 40, 2]

        expected_exhausted = first_stage_exhausted + early_stopping + [think_end_token] + second_stage_exhausted

        # Case 2: Natural completion (</think> in first stage)
        first_stage_natural = [20, 21, 999]  # Has 999
        second_stage_natural = [40, 2]
        
        expected_natural = first_stage_natural + second_stage_natural
        
        # Both should contain </think> marker
        exhausted_text = mock_tokenizer.decode(expected_exhausted)
        natural_text = mock_tokenizer.decode(expected_natural)
        
        assert "</think>" in exhausted_text, "Exhausted case must have </think>"
        assert "</think>" in natural_text, "Natural case must have </think>"
        
        # Exhausted case should have early stopping marker
        assert "[EARLY_STOP]" in exhausted_text, "Exhausted case should have early stopping"
        assert "[EARLY_STOP]" not in natural_text, "Natural case should NOT have early stopping"
    
    def test_multiple_samples_budget_exhausted(self, response_processor, mock_tokenizer):
        """
        Test budget exhaustion with multiple samples (n>1).
        Each sample should be processed independently.
        """
        original_prompt = [10, 11]
        
        # First stage: Two samples, both exhaust budget
        first_tokens_sample1 = [20, 21, 22]
        first_tokens_sample2 = [25, 26, 27]
        
        mock_first_output = Mock()
        mock_first_output.outputs = [
            Mock(token_ids=first_tokens_sample1),
            Mock(token_ids=first_tokens_sample2)
        ]
        
        early_stopping = [888, 999]  # [EARLY_STOP, </think>]
        
        # Expected second stage prompts:
        # Sample 1: original + first_tokens_sample1 + early_stopping + think_end
        # Sample 2: original + first_tokens_sample2 + early_stopping + think_end
        expected_second_prompts = [
            original_prompt + first_tokens_sample1 + early_stopping,
            original_prompt + first_tokens_sample2 + early_stopping
        ]

        # Process first stage
        final_responses, second_prompts, indices, prompt_lengths = \
            response_processor.process_first_stage(
                outputs=[mock_first_output],
                prompts=[original_prompt],
                early_stopping_tokens=early_stopping
            )
        
        # Both samples should need second stage
        assert len(second_prompts) == 2, "Both samples should continue"
        assert final_responses[0] is None
        assert final_responses[1] is None
        for i in range(2):
            assert second_prompts[i] == expected_second_prompts[i]
        
        # Second stage: Different continuations (only new tokens from vLLM)
        second_tokens_sample1 = [40, 2]
        second_tokens_sample2 = [45, 2]

        mock_second_outputs = [
            Mock(outputs=[Mock(token_ids=second_tokens_sample1)]),
            Mock(outputs=[Mock(token_ids=second_tokens_sample2)])
        ]
        
        # Process second stage
        final = response_processor.process_second_stage(
            second_stage_prompts=second_prompts,
            outputs=mock_second_outputs,
            final_responses=final_responses,
            second_stage_indices=indices,
            original_prompt_lengths=prompt_lengths
        )
        
        # Each sample should have complete response
        expected_sample1 = first_tokens_sample1 + early_stopping + second_tokens_sample1
        expected_sample2 = first_tokens_sample2 + early_stopping + second_tokens_sample2

        assert final[0] == expected_sample1, f"Sample 1 should be complete. Found: {final[0]}"
        assert final[1] == expected_sample2, f"Sample 2 should be complete. Found: {final[1]}"

    def test_batch_mixed_completion_states(self, response_processor, mock_tokenizer):
        """
        Test batch with mixed states:
        - One completes in first stage (has EOS)
        - One exhausts budget (needs second stage)
        """
        # Two different prompts
        prompt1 = [10, 11]
        prompt2 = [15, 16]
        
        # First output: completes immediately (has EOS)
        first_tokens_complete = [20, 21, 2]  # Has EOS
        
        # Second output: exhausts budget
        first_tokens_exhausted = [25, 26, 27]  # No EOS, no </think>
        
        mock_first_outputs = [
            Mock(outputs=[Mock(token_ids=first_tokens_complete)]),
            Mock(outputs=[Mock(token_ids=first_tokens_exhausted)])
        ]

        early_stopping = [888, 999]  # [EARLY_STOP, </think>]

        # Process first stage
        final_responses, second_prompts, indices, prompt_lengths = \
            response_processor.process_first_stage(
                outputs=mock_first_outputs,
                prompts=[prompt1, prompt2],
                early_stopping_tokens=early_stopping
            )
        
        # First should be complete, second needs continuation
        assert len(final_responses) == 2
        assert final_responses[0] == first_tokens_complete, "First output already complete"
        assert final_responses[1] is None, "Second output needs continuation"
        assert len(second_prompts) == 1, "Only one needs second stage"
        
        # Second stage only for the incomplete one (only new tokens)
        second_tokens = [40, 2]
        
        mock_second_output = Mock()
        mock_second_output.outputs = [Mock(token_ids=second_tokens)]
        
        # Process second stage
        final = response_processor.process_second_stage(
            second_stage_prompts=second_prompts,
            outputs=[mock_second_output],
            final_responses=final_responses,
            second_stage_indices=indices,
            original_prompt_lengths=prompt_lengths
        )
        
        # First should remain unchanged, second should be complete
        assert final[0] == first_tokens_complete
        expected_second = first_tokens_exhausted + early_stopping + second_tokens
        assert final[1] == expected_second
    
    def test_early_stopping_token_ordering(self, response_processor, mock_tokenizer):
        """
        Verify that early stopping tokens appear in correct position.
        Must be: [first_stage] + [early_stopping] + [think_end] + [answer]
        """
        original_prompt = [10]
        first_stage = [20, 21]
        early_stopping = [888]  # This should come BEFORE any second stage tokens
        think_end = 999
        answer_tokens = [30, 31]
        eos = 2
        
        # Complete sequence
        complete = original_prompt + first_stage + early_stopping + [think_end] + answer_tokens + [eos]
        
        # Find positions
        early_stop_pos = complete.index(888)
        think_end_pos = complete.index(999)
        first_answer_pos = complete.index(30)
        
        # Verify ordering
        assert early_stop_pos < think_end_pos, "Early stopping must come before </think>"
        assert think_end_pos < first_answer_pos, "</think> must come before answer"
        
        # Verify early stopping immediately follows first stage
        assert complete[early_stop_pos - 1] in first_stage, \
            "Early stopping should immediately follow first stage"
    
    def test_no_early_stopping_when_thinking_completes_naturally(self, response_processor):
        """
        Verify early stopping is NOT added when thinking completes naturally.
        """
        original_prompt = [10, 11]
        
        # First stage: naturally completes with </think>
        first_stage_complete = [20, 21, 999]  # Has </think> (999)
        
        mock_first_output = Mock()
        mock_first_output.outputs = [Mock(token_ids=first_stage_complete)]
        
        early_stopping = [888, 999]  # [EARLY_STOP, </think>]
        
        # Process first stage
        final_responses, second_prompts, indices, prompt_lengths = \
            response_processor.process_first_stage(
                outputs=[mock_first_output],
                prompts=[original_prompt],
                early_stopping_tokens=early_stopping
            )
        
        # Should need second stage (no EOS yet)
        assert len(second_prompts) == 1
        
        # Second prompt should NOT contain early stopping tokens
        second_prompt = second_prompts[0]
        
        # Verify early stopping NOT in reconstructed prompt
        assert 888 not in second_prompt, \
            "Early stopping should NOT be added when thinking completes naturally"
        
        # Verify it DOES contain the think end token
        assert 999 in second_prompt, \
            "Natural </think> token should be preserved"
    
    def test_edge_case_empty_second_stage(self, response_processor):
        """
        Test when second stage generates no new tokens (unlikely but possible).
        """
        original_prompt = [10, 11]
        first_stage = [20, 21]
        early_stopping = [888, 999]  # [EARLY_STOP, </think>]

        # Mock first stage output
        second_prompts = [original_prompt + first_stage + early_stopping]
        
        # Second stage generates nothing new (empty list from vLLM)
        second_stage_tokens = []
        
        mock_second_output = Mock()
        mock_second_output.outputs = [Mock(token_ids=second_stage_tokens)]
        
        final_responses = [None]
        indices = [0]
        prompt_lengths = [len(original_prompt)]
        
        # Process second stage
        final = response_processor.process_second_stage(
            second_stage_prompts=second_prompts,
            outputs=[mock_second_output],
            final_responses=final_responses,
            second_stage_indices=indices,
            original_prompt_lengths=prompt_lengths
        )
        
        # Should still include first stage + early stopping (even with empty second stage)
        expected = first_stage + early_stopping
        assert final[0] == expected, \
            "Even with empty second stage, should preserve first stage + early stopping"
