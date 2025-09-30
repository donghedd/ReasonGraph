"""
Module for handling plain text reasoning without visualization.
This allows users to see the raw model output.
"""

def parse_plain_text_response(response_text: str, question: str) -> str:
    """
    Simply pass through the raw model response without any parsing.
    
    Args:
        response_text: The raw response from the API
        question: The original question
    
    Returns:
        The unmodified response text
    """
    return response_text

def create_mermaid_diagram(raw_text: str, config=None) -> None:
    """
    No diagram creation for plain text mode.
    
    Args:
        raw_text: The raw response text
        config: Visualization configuration (not used)
    
    Returns:
        None as no visualization is needed
    """
    return None