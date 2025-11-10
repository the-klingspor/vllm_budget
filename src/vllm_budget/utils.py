"""
Utility functions for vllm_budget library.
"""


def get_default_early_stopping_text() -> str:
    """
    Get default early stopping text.
    
    Returns:
        Default early stopping text string.
    """
    return "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"


def get_default_think_end_token() -> str:
    """
    Get default think end token.
    
    Returns:
        Default think end token string.
    """
    return "</think>"