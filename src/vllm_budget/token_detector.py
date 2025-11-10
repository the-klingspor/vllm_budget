"""
Token detection module for vllm_budget library.
"""

from typing import List, Optional, Any


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
        self.tokenizer = tokenizer
        self.think_end_token = think_end_token
        self._think_end_token_id = think_end_token_id
        
        # Cache EOS token ID
        self._eos_token_id = None
        
        # If think_end_token provided but no ID, try to encode it
        if self.think_end_token and self._think_end_token_id is None:
            try:
                encoded = self.tokenizer.encode(
                    self.think_end_token,
                    add_special_tokens=False
                )
                # Take the last token if multiple tokens
                if encoded:
                    self._think_end_token_id = encoded[-1] if isinstance(encoded, list) else encoded
            except:
                # If encoding fails, keep as None
                pass
    
    def get_eos_token_id(self) -> int:
        """
        Get EOS token ID from tokenizer.
        
        Returns:
            EOS token ID.
        """
        if self._eos_token_id is None:
            # Try common attribute names
            if hasattr(self.tokenizer, 'eos_token_id'):
                self._eos_token_id = self.tokenizer.eos_token_id
            elif hasattr(self.tokenizer, 'eos_id'):
                self._eos_token_id = self.tokenizer.eos_id
            else:
                # Fallback: try to get from special tokens
                if hasattr(self.tokenizer, 'special_tokens'):
                    self._eos_token_id = self.tokenizer.special_tokens.get('eos', 2)
                else:
                    self._eos_token_id = 2  # Common default
        
        return self._eos_token_id
    
    def get_think_end_token_id(self) -> Optional[int]:
        """
        Get think end token ID.
        
        Returns:
            Think end token ID if configured, None otherwise.
        """
        return self._think_end_token_id
    
    def encode_early_stopping_text(self, text: str) -> List[int]:
        """
        Encode early stopping text to token IDs.
        
        Args:
            text: Early stopping text to encode.
            
        Returns:
            List of token IDs.
        """
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Handle different tokenizer return types
        if hasattr(encoded, 'tolist'):
            # PyTorch tensor
            return encoded.tolist()
        elif isinstance(encoded, list):
            return encoded
        else:
            # Single value or other type
            return [encoded] if isinstance(encoded, int) else list(encoded)
    
    def has_eos_token(self, token_ids: List[int]) -> bool:
        """
        Check if token sequence contains EOS token.
        
        Args:
            token_ids: List of token IDs to check.
            
        Returns:
            True if EOS token is present.
        """
        if not token_ids:
            return False
        
        eos_id = self.get_eos_token_id()
        return eos_id in token_ids
    
    def has_think_end_token(self, token_ids: List[int]) -> bool:
        """
        Check if token sequence contains think end token.
        
        Args:
            token_ids: List of token IDs to check.
            
        Returns:
            True if think end token is present.
        """
        if not token_ids:
            return False
        
        think_id = self.get_think_end_token_id()
        if think_id is None:
            return False
        
        return think_id in token_ids