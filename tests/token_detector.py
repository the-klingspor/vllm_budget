from fixtures import token_detector
from vllm_budget.token_detector import TokenDetector


class TestTokenDetector:
    """Tests for TokenDetector."""
    
    def test_get_eos_token_id(self, token_detector):
        """Test retrieval of EOS token ID."""
        eos_id = token_detector.get_eos_token_id()
        assert eos_id == 2
    
    def test_get_think_end_token_id(self, token_detector):
        """Test retrieval of think end token ID."""
        think_id = token_detector.get_think_end_token_id()
        assert think_id == 999
    
    def test_has_eos_token_present(self, token_detector):
        """Test detection when EOS token is present."""
        token_ids = [1, 5, 10, 2, 15]
        assert token_detector.has_eos_token(token_ids) is True
    
    def test_has_eos_token_absent(self, token_detector):
        """Test detection when EOS token is absent."""
        token_ids = [1, 5, 10, 15]
        assert token_detector.has_eos_token(token_ids) is False
    
    def test_has_think_end_token_present(self, token_detector):
        """Test detection when think end token is present."""
        token_ids = [1, 5, 999, 15]
        assert token_detector.has_think_end_token(token_ids) is True
    
    def test_has_think_end_token_absent(self, token_detector):
        """Test detection when think end token is absent."""
        token_ids = [1, 5, 10, 15]
        assert token_detector.has_think_end_token(token_ids) is False
    
    def test_encode_early_stopping_text(self, token_detector):
        """Test encoding of early stopping text."""
        text = "Stop thinking"
        encoded = token_detector.encode_early_stopping_text(text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0
    
    def test_empty_token_list(self, token_detector):
        """Test token detection with empty list."""
        assert token_detector.has_eos_token([]) is False
        assert token_detector.has_think_end_token([]) is False
    
    def test_single_token_eos(self, token_detector):
        """Test with single EOS token."""
        assert token_detector.has_eos_token([2]) is True
    
    def test_no_think_token_configured(self, mock_tokenizer):
        """Test detector without think end token configured."""
        detector = TokenDetector(tokenizer=mock_tokenizer)
        assert detector.get_think_end_token_id() is None
        assert detector.has_think_end_token([1, 2, 3]) is False