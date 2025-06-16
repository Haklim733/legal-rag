import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import json

# Import the module to test
from docs.extract import (
    create_download_directory,
    get_soup,
    get_case_links,
    download_pdf,
    DOWNLOAD_DIR,
    BASE_URL
)

# Test data
MOCK_HTML = """
<html>
    <div id="main-content">
        <a href="/cases/federal/us/123/">Case 1</a>
        <a href="/cases/federal/us/456/">Case 2</a>
        <a href="/cases/federal/us/2024/">2024</a>
    </div>
    <a href="https://example.com/some-other-link">External Link</a>
</html>
"""

MOCK_CASE_HTML = """
<html>
    <a href="case123.pdf">Download PDF</a>
    <h1>Case Title</h1>
</html>
"""

@pytest.fixture
def mock_requests_get():
    with patch('requests.get') as mock_get:
        yield mock_get

@pytest.fixture
def mock_os_makedirs():
    with patch('os.makedirs') as mock_mkdir:
        yield mock_mkdir

def test_create_download_directory(mock_os_makedirs):
    """Test that the download directory is created if it doesn't exist."""
    create_download_directory()
    mock_os_makedirs.assert_called_once_with(DOWNLOAD_DIR, exist_ok=True)

def test_get_soup_success(mock_requests_get):
    """Test successful HTML parsing."""
    mock_response = MagicMock()
    mock_response.text = "<html><body>Test</body></html>"
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response
    
    soup = get_soup("http://example.com")
    assert soup is not None
    assert soup.find('body').text == "Test"

def test_get_soup_failure(mock_requests_get):
    """Test handling of request errors."""
    mock_requests_get.side_effect = Exception("Connection error")
    soup = get_soup("http://example.com")
    assert soup is None

def test_get_case_links(mock_requests_get):
    """Test extraction of case links from HTML."""
    mock_response = MagicMock()
    mock_response.text = MOCK_HTML
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response
    
    links = get_case_links()
    assert len(links) == 2
    assert all(link.startswith(BASE_URL) for link in links)
    assert "/123/" in links[0]
    assert "/456/" in links[1]

def test_get_case_links_with_year(mock_requests_get):
    """Test filtering of case links by year."""
    test_year = 2023
    mock_response = MagicMock()
    mock_response.text = MOCK_HTML
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response
    
    get_case_links(test_year)
    mock_requests_get.assert_called_once_with(f"{BASE_URL}{test_year}/", headers=ANY, timeout=10)

@patch('builtins.open', new_callable=mock_open)
@patch('requests.get')
def test_download_pdf_success(mock_get, mock_file, tmp_path):
    """Test successful PDF download."""
    # Setup mock response for case page
    case_response = MagicMock()
    case_response.text = MOCK_CASE_HTML
    case_response.raise_for_status.return_value = None
    
    # Setup mock response for PDF download
    pdf_response = MagicMock()
    pdf_response.raise_for_status.return_value = None
    pdf_response.iter_content.return_value = [b'%PDF-1.4 ', b'test content']
    
    # Make mock_get return different responses for different URLs
    mock_get.side_effect = [case_response, pdf_response]
    
    # Test the function
    test_url = f"{BASE_URL}123/"
    with patch('pathlib.Path.exists', return_value=False):
        download_pdf(test_url)
    
    # Verify the PDF was downloaded
    expected_path = DOWNLOAD_DIR / "123.pdf"
    mock_file.assert_called_once_with(expected_path, 'wb')

@patch('builtins.print')
def test_download_pdf_exists(mock_print, tmp_path):
    """Test skipping of existing PDFs."""
    test_url = f"{BASE_URL}123/"
    with patch('pathlib.Path.exists', return_value=True):
        download_pdf(test_url)
    
    # Verify the function printed the skip message
    mock_print.assert_called_with("Skipping tests/123.pdf - already exists")
