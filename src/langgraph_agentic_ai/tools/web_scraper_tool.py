import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool
def scrape_webpage(url: str) -> str:
    """Fetch and extract text content from a webpage URL. Useful for reading articles, documentation, or any web page."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Truncate to avoid token overflow
        return text[:3000] if text else "No text content found on the page."
    except requests.exceptions.Timeout:
        return f"Error: Request timed out for URL: {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)[:200]}"
    except Exception as e:
        return f"Error parsing page: {str(e)[:200]}"
