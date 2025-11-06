from typing import List, Optional, Union, Any


# ============================================================================
# Main Wrapper
# ============================================================================

class ThinkingBudgetLLM:
    """
    Wrapper around vLLM's LLM class with thinking budget support.
    
    Enables two-stage generation where thinking phase has a token budget,
    and automatically continues generation if budget is exhausted.
    """
    
    def __init__(
        self,
        model: str,
        thinking_budget: Optional[int] = None,
        early_stopping_text: Optional[str] = None,
        think_end_token: Optional[str] = None,
        think_end_token_id: Optional[int] = None,
        **vllm_kwargs
    ):
        """
        Initialize ThinkingBudgetLLM.
        
        Args:
            model: Model name or path.
            thinking_budget: Default thinking budget in tokens.
            early_stopping_text: Text to insert when thinking budget exhausted.
            think_end_token: String representation of think end token.
            think_end_token_id: Token ID for think end token.
            **vllm_kwargs: Additional arguments passed to vLLM LLM constructor.
        """
        pass
    
    @classmethod
    def from_vllm(
        cls,
        vllm_instance: Any,
        thinking_budget: Optional[int] = None,
        early_stopping_text: Optional[str] = None,
        think_end_token: Optional[str] = None,
        think_end_token_id: Optional[int] = None
    ) -> 'ThinkingBudgetLLM':
        """
        Create ThinkingBudgetLLM from existing vLLM instance.
        
        Args:
            vllm_instance: Existing vLLM LLM instance.
            thinking_budget: Default thinking budget in tokens.
            early_stopping_text: Text to insert when thinking budget exhausted.
            think_end_token: String representation of think end token.
            think_end_token_id: Token ID for think end token.
            
        Returns:
            ThinkingBudgetLLM instance wrapping the vLLM instance.
        """
        pass
    
    def generate(
        self,
        prompts: Union[str, List[str], List[List[int]]],
        sampling_params: Any,
        thinking_budget: Optional[int] = None,
        use_tqdm: bool = True
    ) -> List[Any]:
        """
        Generate responses with thinking budget.
        
        If thinking_budget is None, performs standard vLLM generation.
        Otherwise, uses two-stage generation with budget constraint.
        
        Args:
            prompts: Input prompts (strings or token IDs).
            sampling_params: vLLM SamplingParams instance.
            thinking_budget: Override default thinking budget for this call.
            use_tqdm: Whether to show progress bar.
            
        Returns:
            List of vLLM RequestOutput objects.
        """
        pass
    
    def _generate_with_thinking_budget(
        self,
        prompts: Union[str, List[str], List[List[int]]],
        sampling_params: Any,
        thinking_budget: int,
        use_tqdm: bool
    ) -> List[Any]:
        """
        Internal method implementing two-stage generation logic.
        
        Args:
            prompts: Input prompts.
            sampling_params: Sampling parameters.
            thinking_budget: Thinking budget in tokens.
            use_tqdm: Whether to show progress bar.
            
        Returns:
            List of modified vLLM RequestOutput objects.
        """
        pass
    
    def _normalize_prompts(
        self,
        prompts: Union[str, List[str], List[List[int]]]
    ) -> List[Union[str, List[int]]]:
        """
        Normalize prompts to list format.
        
        Args:
            prompts: Input prompts in various formats.
            
        Returns:
            List of prompts.
        """
        pass
    
    def _copy_sampling_params(self, sampling_params: Any) -> Any:
        """
        Create a copy of sampling parameters.
        
        Args:
            sampling_params: Original sampling parameters.
            
        Returns:
            Copy of sampling parameters.
        """
        pass
    
    def _create_modified_output(
        self,
        original_output: Any,
        new_token_ids: List[int]
    ) -> Any:
        """
        Create modified vLLM output with new token IDs.
        
        Args:
            original_output: Original vLLM RequestOutput.
            new_token_ids: New token ID sequence.
            
        Returns:
            Modified RequestOutput object.
        """
        pass