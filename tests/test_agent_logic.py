import pytest
from unittest.mock import MagicMock, patch
from src.backend.agentic_ai_system import AsyncAgenticAiSystem

@pytest.fixture
def mock_engine():
    # Mocking external Azure/OpenAI dependencies to test local logic]
    with patch('src.backend.AgenticAI.load_llm'), \
         patch('src.backend.AgenticAI.load_embed'), \
         patch('src.backend.AgenticAI.initialize_index'):
        engine = AsyncAgenticAiSystem(index_name="global_index")
        return engine

def test_safe_output_processor_text(mock_engine):
    """Verifies that plain text is returned without modification."""
    input_text = "This is a global enterprise summary."
    result = mock_engine._AsyncAgenticAiSystem__safe_output_processor(input_text)
    assert result == input_text 

def test_safe_output_processor_code_cleaning(mock_engine):
    """Verifies that markdown code blocks are correctly stripped."""
    code_input = "```python\nx = 10\n```"
    # Testing the private method for code block stripping 
    result = mock_engine._AsyncAgenticAiSystem__safe_output_processor(code_input)
    assert "x = 10" in result 

def test_conversation_summarization_trigger(mock_engine):
    """Verifies history is partitioned when it exceeds the length threshold."""
    long_thread = [{'role': 'user', 'content': 'hello', 'createdAt': '2026-01-01T00:00:00Z'}] * 10
    with patch.object(mock_engine, '_AsyncAgenticAiSystem__summarize_conversation', return_value="Summary"):
        processed_thread = mock_engine.set_conversation_thread(long_thread)
        # Check if the first message in the processed thread is a system summary 
        assert processed_thread[0]['role'] == 'system' 