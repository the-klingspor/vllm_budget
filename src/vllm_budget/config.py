from typing import Optional
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ThinkingBudgetConfig:
    """Configuration for thinking budget generation."""
    
    thinking_budget: int
    early_stopping_text: str
    think_end_token: Optional[str] = None
    think_end_token_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        pass
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        pass
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ThinkingBudgetConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters.
            
        Returns:
            ThinkingBudgetConfig instance.
        """
        pass