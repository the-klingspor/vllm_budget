import pytest

from vllm_budget.thinking_budget_config import ThinkingBudgetConfig

class TestThinkingBudgetConfig:
    """Tests for ThinkingBudgetConfig."""
    
    def test_valid_config(self):
        """Test creation of valid configuration."""
        config = ThinkingBudgetConfig(
            thinking_budget=100,
            early_stopping_text="Stop here",
            think_end_token="</think>"
        )
        assert config.thinking_budget == 100
        assert config.early_stopping_text == "Stop here"
    
    def test_zero_thinking_budget(self):
        """Test that zero thinking budget raises error."""
        with pytest.raises(ValueError):
            config = ThinkingBudgetConfig(
                thinking_budget=0,
                early_stopping_text="Stop"
            )
            config.validate()
    
    def test_negative_thinking_budget(self):
        """Test that negative thinking budget raises error."""
        with pytest.raises(ValueError):
            config = ThinkingBudgetConfig(
                thinking_budget=-10,
                early_stopping_text="Stop"
            )
            config.validate()
    
    def test_empty_early_stopping_text(self):
        """Test that empty early stopping text raises error."""
        with pytest.raises(ValueError):
            config = ThinkingBudgetConfig(
                thinking_budget=100,
                early_stopping_text=""
            )
            config.validate()
    
    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "thinking_budget": 100,
            "early_stopping_text": "Stop",
            "think_end_token": "</think>"
        }
        config = ThinkingBudgetConfig.from_dict(config_dict)
        assert config.thinking_budget == 100
        assert config.think_end_token == "</think>"
    
    def test_missing_required_fields(self):
        """Test that missing required fields raises error."""
        with pytest.raises((TypeError, KeyError)):
            ThinkingBudgetConfig.from_dict({"thinking_budget": 100})