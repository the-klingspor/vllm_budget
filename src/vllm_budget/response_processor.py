"""
Response processing module for vllm_budget library.
"""

from typing import List, Optional, Union, Any, Tuple

from vllm_budget.token_detector import TokenDetector


class ResponseProcessor:
    """Processes multi-stage generation responses."""
    
    def __init__(self, tokenizer: Any, token_detector: TokenDetector):
        """
        Initialize response processor.
        
        Args:
            tokenizer: Tokenizer instance.
            token_detector: TokenDetector instance.
        """
        self.tokenizer = tokenizer
        self.token_detector = token_detector
    
    def process_first_stage(
        self,
        outputs: List[Any],
        prompts: List[Union[str, List[int]]],
        early_stopping_tokens: List[int]
    ) -> Tuple[List[Optional[List[int]]], List[Union[str, List[int]]], List[int], List[int]]:
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
        final_responses = []
        second_stage_prompts = []
        second_stage_indices = []
        original_prompt_lengths = []
        
        for batch_idx, output in enumerate(outputs):
            # Get corresponding prompt
            prompt = prompts[batch_idx]
            
            for sample_id in range(len(output.outputs)):
                first_tokens = output.outputs[sample_id].token_ids
                
                # Check if generation already finished (hit end token)
                if self.token_detector.has_eos_token(first_tokens):
                    final_responses.append(first_tokens)
                    continue
                
                # Check if thinking phase completed naturally (found </think> token)
                if self.token_detector.has_think_end_token(first_tokens):
                    # Thinking completed, prepare for second stage
                    combined_prompt = self.reconstruct_prompt(prompt, first_tokens)
                    
                    second_stage_prompts.append(combined_prompt)
                    second_stage_indices.append(len(final_responses))
                    original_prompt_lengths.append(self.get_prompt_token_length(prompt))
                    final_responses.append(None)  # Placeholder
                else:
                    # Thinking budget exhausted, add early stopping text
                    combined_prompt = self.reconstruct_prompt(
                        prompt, first_tokens, early_stopping_tokens
                    )
                    
                    second_stage_prompts.append(combined_prompt)
                    second_stage_indices.append(len(final_responses))
                    original_prompt_lengths.append(self.get_prompt_token_length(prompt))
                    final_responses.append(None)  # Placeholder
        
        return final_responses, second_stage_prompts, second_stage_indices, original_prompt_lengths
    
    def process_second_stage(
        self,
        second_stage_prompts: List[Union[str, List[int]]],
        outputs: List[Any],
        final_responses: List[Optional[List[int]]],
        second_stage_indices: List[int],
        original_prompt_lengths: List[int]
    ) -> List[List[int]]:
        """
        Process second stage generation outputs and merge with first stage.
        
        Args:
            second_stage_prompts: Prompts used for second stage generation, including first_response, early_stopping and </think>.
            outputs: vLLM generation outputs from second stage.
            final_responses: Response list with placeholders from first stage.
            second_stage_indices: Indices mapping second stage to final responses.
            original_prompt_lengths: Original prompt lengths for token extraction.
            
        Returns:
            Complete list of final token sequences.
        """
        for i, output in enumerate(outputs):
            for sample_id in range(len(output.outputs)):

                prompt = second_stage_prompts[i]
                if isinstance(prompt, str):
                    prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

                else:
                    prompt_tokens = prompt

                
                # Combine second stage prompt tokens with generated tokens
                complete_tokens = prompt_tokens + output.outputs[sample_id].token_ids

                # Extract only the new tokens (remove original prompt)
                original_length = original_prompt_lengths[i]
                final_tokens = complete_tokens[original_length:]
                
                # Update final response
                response_idx = second_stage_indices[i]
                final_responses[response_idx] = final_tokens
        
        return final_responses
    
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
        if isinstance(prompt, str):
            # String format: encode prompt, combine, decode back
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            
            # Handle different tokenizer return types
            if hasattr(prompt_tokens, 'tolist'):
                prompt_tokens = prompt_tokens.tolist()
            elif not isinstance(prompt_tokens, list):
                prompt_tokens = [prompt_tokens] if isinstance(prompt_tokens, int) else list(prompt_tokens)
            
            # Combine tokens
            combined_tokens = prompt_tokens + first_tokens
            if additional_tokens:
                combined_tokens += additional_tokens
            
            # Decode back to string
            return self.tokenizer.decode(combined_tokens, skip_special_tokens=False)
        else:
            # Token ID format: direct concatenation
            combined_tokens = list(prompt) + first_tokens
            if additional_tokens:
                combined_tokens += additional_tokens
            return combined_tokens
    
    def get_prompt_token_length(self, prompt: Union[str, List[int]]) -> int:
        """
        Get token length of prompt.
        
        Args:
            prompt: Prompt as string or token IDs.
            
        Returns:
            Number of tokens in prompt.
        """
        if isinstance(prompt, str):
            encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
            if hasattr(encoded, '__len__'):
                return len(encoded)
            else:
                return 1  # Single token
        else:
            return len(prompt)