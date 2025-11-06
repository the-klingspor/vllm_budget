from typing import List, Optional, Union, Any

from vllm_budget.token_detector import TokenDetector

# ============================================================================
# Response Processing
# ============================================================================

class ResponseProcessor:
    """Processes multi-stage generation responses."""
    
    def __init__(self, tokenizer: Any, token_detector: TokenDetector):
        """
        Initialize response processor.
        
        Args:
            tokenizer: Tokenizer instance.
            token_detector: TokenDetector instance.
        """
        pass
    
    def process_first_stage(
        self,
        outputs: List[Any],
        prompts: List[Union[str, List[int]]],
        early_stopping_tokens: List[int]
    ) -> tuple:
        """
        Process first stage generation outputs.
        
        Args:
            outputs: vLLM generation outputs from first stage.
            prompts: Original input prompts.
            early_stopping_tokens: Token IDs for early stopping text.
            
        Returns:
            Tuple of (final_responses, second_stage_prompts, second_stage_indices, 
                     original_prompt_lengths).
        """
        pass
    
    def process_second_stage(
        self,
        outputs: List[Any],
        final_responses: List[Optional[List[int]]],
        second_stage_indices: List[int],
        original_prompt_lengths: List[int]
    ) -> List[List[int]]:
        """
        Process second stage generation outputs and merge with first stage.
        
        Args:
            outputs: vLLM generation outputs from second stage.
            final_responses: Response list with placeholders from first stage.
            second_stage_indices: Indices mapping second stage to final responses.
            original_prompt_lengths: Original prompt lengths for token extraction.
            
        Returns:
            Complete list of final token sequences.
        """
        pass
    
    def reconstruct_prompt(
        self,
        prompt: Union[str, List[int]],
        first_tokens: List[int],
        additional_tokens: Optional[List[int]] = None
    ) -> Union[str, List[int]]:
        """
        Reconstruct prompt with generated tokens for continuation.
        
        Args:
            prompt: Original prompt (string or token IDs).
            first_tokens: Token IDs from first generation stage.
            additional_tokens: Optional additional tokens to append.
            
        Returns:
            Combined prompt in same format as input.
        """
        pass
    
    def get_prompt_token_length(self, prompt: Union[str, List[int]]) -> int:
        """
        Get token length of prompt.
        
        Args:
            prompt: Prompt as string or token IDs.
            
        Returns:
            Number of tokens in prompt.
        """
        pass