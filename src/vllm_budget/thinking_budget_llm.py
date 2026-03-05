"""
Main wrapper module for vllm_budget library.
"""

from typing import Dict, List, Optional, Union, Any
from copy import deepcopy

from vllm import LLM
from vllm_budget.config import ThinkingBudgetConfig
from vllm_budget.token_detector import TokenDetector
from vllm_budget.response_processor import ResponseProcessor
from vllm_budget.utils import get_default_early_stopping_text, get_default_think_end_token


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
        # Initialize vLLM engine
        self.llm = LLM(model=model, **vllm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        
        # Set defaults
        self.default_thinking_budget = thinking_budget
        self.early_stopping_text = early_stopping_text or get_default_early_stopping_text()
        
        # Initialize token detector
        self.token_detector = TokenDetector(
            tokenizer=self.tokenizer,
            think_end_token=think_end_token or get_default_think_end_token(),
            think_end_token_id=think_end_token_id
        )
        
        # Initialize response processor
        self.response_processor = ResponseProcessor(
            tokenizer=self.tokenizer,
            token_detector=self.token_detector
        )
        
        # Pre-encode early stopping text
        self._early_stopping_tokens = self.token_detector.encode_early_stopping_text(
            self.early_stopping_text
        )
    
    @classmethod
    def from_vllm(
        cls,
        vllm_instance: LLM,
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
        # Create instance without initializing a new LLM
        instance = cls.__new__(cls)
        instance.llm = vllm_instance
        instance.tokenizer = vllm_instance.get_tokenizer()
        
        instance.default_thinking_budget = thinking_budget
        instance.early_stopping_text = early_stopping_text or get_default_early_stopping_text()
        
        instance.token_detector = TokenDetector(
            tokenizer=instance.tokenizer,
            think_end_token=think_end_token or get_default_think_end_token(),
            think_end_token_id=think_end_token_id
        )
        
        instance.response_processor = ResponseProcessor(
            tokenizer=instance.tokenizer,
            token_detector=instance.token_detector
        )
        
        instance._early_stopping_tokens = instance.token_detector.encode_early_stopping_text(
            instance.early_stopping_text
        )
        
        return instance
    
    def generate(
        self,
        prompts: Union[
            str,                           # Single text prompt
            List[str],                     # List of text prompts
            Dict[str, List[int]],          # Single TokensPrompt
            List[Dict[str, List[int]]],    # List of TokensPrompt
            List[List[int]],               # List of token ID prompts
        ],
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
        # Determine which budget to use
        budget = thinking_budget if thinking_budget is not None else self.default_thinking_budget
        
        # If no budget specified, use standard generation
        if budget is None:
            return self.llm.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm
            )
        
        # Use thinking budget generation
        return self._generate_with_thinking_budget(
            prompts=prompts,
            sampling_params=sampling_params,
            thinking_budget=budget,
            use_tqdm=use_tqdm
        )
    
    def _generate_with_thinking_budget(
        self,
        prompts: Union[
            str,                           # Single text prompt
            List[str],                     # List of text prompts
            Dict[str, List[int]],          # Single TokensPrompt
            List[Dict[str, List[int]]],    # List of TokensPrompt
            List[List[int]],               # List of token ID prompts
        ],
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
        # Normalize prompts to list format
        normalized_prompts = self._normalize_prompts(prompts)
        
        # Stage 1: Generate up to thinking budget
        first_sampling_params = self._copy_sampling_params(sampling_params)
        first_sampling_params.max_tokens = thinking_budget
        
        first_outputs = self.llm.generate(
            prompts=normalized_prompts,
            sampling_params=first_sampling_params,
            use_tqdm=use_tqdm
        )
        
        # Process first stage results
        final_responses, second_stage_prompts, second_stage_indices, original_prompt_lengths = \
            self.response_processor.process_first_stage(
                outputs=first_outputs,
                prompts=normalized_prompts,
                early_stopping_tokens=self._early_stopping_tokens
            )
        
        # Stage 2: Complete generation for incomplete outputs
        if second_stage_prompts:
            # Calculate remaining token budget
            remaining_tokens = sampling_params.max_tokens - thinking_budget
            if remaining_tokens <= 0:
                remaining_tokens = sampling_params.max_tokens  # Fallback
            
            second_sampling_params = self._copy_sampling_params(sampling_params)
            second_sampling_params.max_tokens = remaining_tokens
            
            second_outputs = self.llm.generate(
                prompts=second_stage_prompts,
                sampling_params=second_sampling_params,
                use_tqdm=use_tqdm
            )
            
            # Process second stage results
            final_responses = self.response_processor.process_second_stage(
                second_stage_prompts=second_stage_prompts,
                outputs=second_outputs,
                final_responses=final_responses,
                second_stage_indices=second_stage_indices,
                original_prompt_lengths=original_prompt_lengths
            )
        
        # Convert final responses back to RequestOutput format
        return self._create_output_objects(first_outputs, final_responses)
    
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
        if isinstance(prompts, str):
            return [prompts]
        elif isinstance(prompts, list):
            if all(isinstance(p, str) for p in prompts):
                return prompts
            elif all(isinstance(p, dict) for p in prompts):
                return prompts
            elif all(isinstance(p, list) for p in prompts):
                return [{"prompt_token_ids": p} for p in prompts]
        elif isinstance(prompts, dict):
            return [prompts]
        else:
            return [prompts]
    
    def _copy_sampling_params(self, sampling_params: Any) -> Any:
        """
        Create a copy of sampling parameters.
        
        Args:
            sampling_params: Original sampling parameters.
            
        Returns:
            Copy of sampling parameters.
        """
        # Try different copy methods
        if hasattr(sampling_params, 'copy'):
            return sampling_params.copy()
        else:
            # Use deepcopy as fallback
            return deepcopy(sampling_params)
    
    def _create_output_objects(
        self,
        original_outputs: List[Any],
        final_token_sequences: List[List[int]]
    ) -> List[Any]:
        """
        Create modified vLLM output objects with final token sequences.
        
        Args:
            original_outputs: Original vLLM RequestOutput objects.
            final_token_sequences: Final token ID sequences.
            
        Returns:
            List of modified RequestOutput objects.
        """
        # Create new outputs based on original structure
        modified_outputs = []
        token_idx = 0
        
        for output in original_outputs:
            # Create a copy of the output
            modified_output = deepcopy(output)
            
            # Update token IDs for each sample
            for sample_id in range(len(modified_output.outputs)):
                if token_idx < len(final_token_sequences):
                    modified_output.outputs[sample_id].token_ids = final_token_sequences[token_idx]
                    
                    # Update text representation if available
                    if hasattr(modified_output.outputs[sample_id], 'text'):
                        modified_output.outputs[sample_id].text = self.tokenizer.decode(
                            final_token_sequences[token_idx],
                            skip_special_tokens=False
                        )
                    
                    token_idx += 1
            
            modified_outputs.append(modified_output)
        
        return modified_outputs