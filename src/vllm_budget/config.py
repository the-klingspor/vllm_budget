"""
Configuration module for vllm_budget library.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class ThinkingBudgetConfig:
    """Configuration for thinking budget generation."""
    
    thinking_budget: int
    early_stopping_text: str
    think_end_token: Optional[str] = None
    think_end_token_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        if self.thinking_budget <= 0:
            raise ValueError(
                f"thinking_budget must be positive, got {self.thinking_budget}"
            )
        
        if not self.early_stopping_text or len(self.early_stopping_text.strip()) == 0:
            raise ValueError("early_stopping_text cannot be empty")
        
        if self.think_end_token_id is not None and self.think_end_token_id < 0:
            raise ValueError(
                f"think_end_token_id must be non-negative, got {self.think_end_token_id}"
            )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ThinkingBudgetConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters.
            
        Returns:
            ThinkingBudgetConfig instance.
            
        Raises:
            KeyError: If required fields are missing.
            TypeError: If field types are incorrect.
        """
        required_fields = ['thinking_budget', 'early_stopping_text']
        for field in required_fields:
            if field not in config_dict:
                raise KeyError(f"Required field '{field}' missing from config_dict")
        
        return cls(**config_dict)