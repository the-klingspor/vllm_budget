from vllm_budget.utils import (
    get_default_early_stopping_text,
    get_default_think_end_token
)

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_default_early_stopping_text(self):
        """Test default early stopping text retrieval."""
        text = get_default_early_stopping_text()
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_get_default_think_end_token(self):
        """Test default think end token retrieval."""
        token = get_default_think_end_token()
        assert isinstance(token, str)
        assert len(token) > 0