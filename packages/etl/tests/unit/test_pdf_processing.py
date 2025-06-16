import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from docs.process_pdfs import extract_text_from_pdf, process_pdf_directory, partition_pdf

# Get the path to the test PDF
TEST_PDF_PATH = Path(__file__).parent.parent.parent / "assets" / "supreme-court" / "2025" / "24A1007_AARPvTrump_20250516.pdf"

def test_extract_text_from_pdf():
    """Test that we can extract text from a PDF file."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")
    
    try:
        # Test text extraction
        text, elements = extract_text_from_pdf(TEST_PDF_PATH)
        
        # Basic assertions
        assert isinstance(text, str), "Expected text to be a string"
        assert len(text) > 0, "Expected non-empty text"
        assert isinstance(elements, list), "Expected elements to be a list"
        
        # Skip elements check if no elements were extracted (might be an issue with test file)
        if len(elements) == 0:
            pytest.skip("No elements extracted from PDF - check if the test file has extractable text")
        
        # Test that elements have the expected structure
        for element in elements:
            assert "type" in element, "Element missing 'type' field"
            assert "text" in element, "Element missing 'text' field"
            assert isinstance(element["text"], str), "Element text should be a string"
            assert "metadata" in element, "Element missing 'metadata' field"
            assert isinstance(element["metadata"], dict), "Element metadata should be a dictionary"
    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise

@patch('docs.process_pdfs.partition_pdf')
def test_extract_text_from_pdf_error_handling(mock_partition):
    """Test error handling when processing a PDF that raises an error."""
    # Setup mock to raise an exception when partition_pdf is called
    mock_partition.side_effect = Exception("Test error")
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            # Test that the function handles errors properly
            result_text, result_elements = extract_text_from_pdf(temp_path)
            assert result_text == "", "Expected empty string on error"
            assert result_elements == [], "Expected empty list on error"
        finally:
            if temp_path.exists():
                temp_path.unlink()

@patch('docs.process_pdfs.extract_text_from_pdf')
def test_process_pdf_directory(mock_extract):
    """Test processing a directory of PDFs."""
    # Setup mock return value
    mock_extract.return_value = ("Test text", [{"type": "Text", "text": "Test", "metadata": {}}])
    
    # Create a temporary directory with a test PDF
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf = Path(temp_dir) / "test.pdf"
        temp_pdf.touch()  # Create empty file
        
        try:
            # Test processing the directory
            results = process_pdf_directory(Path(temp_dir))
            
            # Verify results
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]["file"] == str(temp_pdf)
            assert results[0]["success"] is True
            assert "text" in results[0]
            assert "elements" in results[0]
        except RuntimeError as e:
            if "poppler" in str(e).lower():
                pytest.skip(f"Skipping test due to missing system dependency: {e}")
            raise

def test_process_pdf_directory_error_handling():
    """Test error handling with a non-directory path."""
    try:
        with pytest.raises(ValueError, match="is not a valid directory"):
            process_pdf_directory(Path("nonexistent_dir"))
    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise

@patch('docs.process_pdfs.extract_text_from_pdf')
def test_process_pdf_directory_empty(mock_extract):
    """Test processing an empty directory."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test processing an empty directory
            results = process_pdf_directory(Path(temp_dir))
            assert results == []
    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise

@patch('docs.process_pdfs.extract_text_from_pdf')
def test_process_pdf_directory_with_errors(mock_extract):
    """Test processing a directory with a PDF that raises an error."""
    try:
        # Setup mock to raise an exception
        mock_extract.side_effect = Exception("Test error")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf = Path(temp_dir) / "test.pdf"
            temp_pdf.touch()  # Create empty file
            
            # Test processing the directory
            results = process_pdf_directory(Path(temp_dir))
            
            # Verify error handling
            assert len(results) == 1
            assert results[0]["file"] == str(temp_pdf)
            assert results[0]["success"] is False
            assert "error" in results[0]
    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise
