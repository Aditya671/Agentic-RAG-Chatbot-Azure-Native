import os
import json
import pytest
from unittest.mock import patch, MagicMock
from src.backend.user_uploaded_file_indexer import UserUploadedFileIndexer

@pytest.fixture
def mock_indexer(tmp_path):
    # Use a temporary directory for index data to keep tests clean
    root_dir = tmp_path / "uploads"
    root_dir.mkdir()
    with patch('src.backend.user_uploaded_file_indexer.load_embed'), \
         patch('src.backend.user_uploaded_file_indexer.load_llm'):
        indexer = UserUploadedFileIndexer(root_dir=str(root_dir), index_name="test_index")
        return indexer, root_dir

def test_should_reindex_new_file(mock_indexer):
    """Verifies that a file not in metadata triggers a reindex."""
    indexer, root_dir = mock_indexer
    new_file = root_dir / "global_report.pdf"
    new_file.write_text("content")

    # Path is not in metadata yet, so it should return True
    assert indexer._UserUploadedFileIndexer__should_reindex(str(new_file)) is True

def test_file_hash_consistency(tmp_path):
    """Verifies the utility correctly computes SHA256 hashes for file integrity."""
    from src.backend.utility import compute_file_hash
    test_file = tmp_path / "data.txt"
    test_file.write_bytes(b"enterprise_data")

    hash1 = compute_file_hash(str(test_file))
    hash2 = compute_file_hash(str(test_file))

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 length