import pytest
from unittest.mock import patch, mock_open, MagicMock, ANY
from pathlib import Path
import json
import requests

# Import the module to test
from src.docs.extract_scotus import (
    create_download_directory,
    get_soup,
    get_case_links,
    download_pdf,
    DOWNLOAD_DIR,
    BASE_URL,
)

# Test data
MOCK_HTML = """
<html>
    <div id="main-content">
        <a href="/cases/federal/us/123">Case 1</a>
        <a href="/cases/federal/us/456">Case 2</a>
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
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_path_mkdir():
    with patch.object(Path, "mkdir") as mock_mkdir:
        yield mock_mkdir


def test_create_download_directory(mock_path_mkdir):
    """Test that the download directory is created if it doesn't exist."""
    create_download_directory()
    mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_get_soup_success(mock_requests_get):
    """Test successful HTML parsing."""
    mock_response = MagicMock()
    mock_response.text = "<html><body>Test</body></html>"
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response

    soup = get_soup("http://example.com")
    assert soup is not None
    assert soup.find("body").text == "Test"


def test_get_soup_failure(mock_requests_get):
    """Test handling of request errors."""
    mock_requests_get.side_effect = requests.RequestException("Connection error")

    with patch("builtins.print") as mock_print:
        soup = get_soup("http://example.com")

    assert soup is None
    mock_print.assert_called_once_with(
        "Error fetching http://example.com: Connection error"
    )


def test_get_case_links(mock_requests_get):
    """Test extraction of case links from HTML."""
    mock_response = MagicMock()
    mock_response.text = MOCK_HTML
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response

    links = get_case_links()
    assert len(links) == 2
    assert all(link.startswith(BASE_URL) for link in links)
    assert "/123" in links[0]
    assert "/456" in links[1]


def test_get_case_links_with_year(mock_requests_get):
    """Test filtering of case links by year."""
    test_year = 2023
    mock_response = MagicMock()
    mock_response.text = MOCK_HTML
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response

    get_case_links(test_year)
    mock_requests_get.assert_called_once_with(
        f"{BASE_URL}{test_year}/", headers=ANY, timeout=10
    )
