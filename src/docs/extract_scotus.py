import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
from typing import List, Optional

# Configuration
BASE_URL = "https://supreme.justia.com/cases/federal/us/"
DOWNLOAD_DIR = Path("supreme_court_pdfs")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def create_download_directory() -> None:
    """Create download directory if it doesn't exist."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_soup(url: str) -> Optional[BeautifulSoup]:
    """Get BeautifulSoup object from URL."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def get_case_links(year: Optional[int] = None) -> List[str]:
    """Get all case links, optionally filtered by year."""
    url = f"{BASE_URL}{str(year)}/" if year else BASE_URL
    soup = get_soup(url)
    if not soup:
        return []

    case_links = []
    for a in soup.select('div#main-content a[href*="/cases/federal/us/"]'):
        href = a.get("href")
        if href and not href.endswith("/"):  # Skip year links
            case_links.append(urljoin(BASE_URL, href))

    return case_links


def download_pdf(case_url: str) -> None:
    """Download PDF for a single case."""
    soup = get_soup(case_url)
    if not soup:
        return

    # Find PDF link (if available)
    pdf_link = soup.find("a", href=re.compile(r"\.pdf$", re.I))
    if not pdf_link:
        print(f"No PDF found for {case_url}")
        return

    pdf_url = urljoin(case_url, pdf_link["href"])
    case_id = case_url.rstrip("/").split("/")[-1]
    filename = DOWNLOAD_DIR / f"{case_id}.pdf"

    if filename.exists():
        print(f"Skipping {filename} - already exists")
        return

    try:
        print(f"Downloading {pdf_url}...")
        response = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=30)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved {filename}")

    except requests.RequestException as e:
        print(f"Error downloading {pdf_url}: {e}")


def main():
    """Main function to download all available Supreme Court case PDFs."""
    create_download_directory()

    # Get all case links
    print("Fetching case links...")
    case_links = get_case_links()

    if not case_links:
        print("No case links found.")
        return

    print(f"Found {len(case_links)} cases. Starting downloads...")

    # Download PDFs with rate limiting
    for i, case_url in enumerate(case_links, 1):
        print(f"\nProcessing case {i}/{len(case_links)}: {case_url}")
        download_pdf(case_url)
        time.sleep(1)  # Be nice to the server


if __name__ == "__main__":
    main()
