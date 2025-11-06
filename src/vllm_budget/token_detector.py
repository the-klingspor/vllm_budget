from typing import List, Optional, Any

# ============================================================================
# Token Detection
# ============================================================================

class TokenDetector:
    """Detects special tokens in generated sequences."""
    
    def __init__(self, tokenizer: Any, think_end_token: Optional[str] = None,
                 think_end_token_id: Optional[int] = None):
        """
        Initialize token detector.
        
        Args:
            tokenizer: Tokenizer instance (e.g., from transformers).
            think_end_token: String representation of think end token.
            think_end_token_id: Token ID for think end token.
        """
        pass
    
    def get_eos_token_id(self) -> int:
        """
        Get EOS token ID from tokenizer.
        
        Returns:
            EOS token ID.
        """
        pass
    
    def get_think_end_token_id(self) -> Optional[int]:
        """
        Get think end token ID.
        
        Returns:
            Think end token ID if configured, None otherwise.
        """
        pass
    
    def encode_early_stopping_text(self, text: str) -> List[int]:
        """
        Encode early stopping text to token IDs.
        
        Args:
            text: Early stopping text to encode.
            
        Returns:
            List of token IDs.
        """
        pass
    
    def has_eos_token(self, token_ids: List[int]) -> bool:
        """
        Check if token sequence contains EOS token.
        
        Args:
            token_ids: List of token IDs to check.
            
        Returns:
            True if EOS token is present.
        """
        pass
    
    def has_think_end_token(self, token_ids: List[int]) -> bool:
        """
        Check if token sequence contains think end token.
        
        Args:
            token_ids: List of token IDs to check.
            
        Returns:
            True if think end token is present.
        """
        pass
