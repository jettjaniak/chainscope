import logging
import os
from typing import Any

from googleapiclient.discovery import build
from zyte_api import ZyteAPI

from chainscope.api_utils.open_ai_utils import generate_oa_response_sync
from chainscope.properties import get_value
from chainscope.typing import *

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
ZYTE_API_KEY = os.getenv("ZYTE_API_KEY")


def google_search(query: str, *, num_results: int = 10) -> list[dict[str, Any]]:
    """
    Perform a Google search using the Custom Search JSON API.
    
    Args:
        query: The search query string
        api_key: Google API key
        cx: Google Custom Search Engine ID
        num_results: Number of results to return (max 10 per request)
        
    Returns:
        List of search results, where each result is a dictionary containing
        keys like 'title', 'link', 'snippet', etc.
    """
    try:
        # Build the service object
        service = build('customsearch', 'v1', developerKey=GOOGLE_SEARCH_API_KEY)
        
        # Execute the search
        result = service.cse().list(
            q=query,
            cx=GOOGLE_SEARCH_ENGINE_ID,
            num=min(num_results, 10)  # API limit is 10 results per request
        ).execute()
        
        # Extract search items
        search_results = result.get('items', [])
        
        return search_results
        
    except Exception as e:
        raise Exception(f"Google search failed: {str(e)}")


def get_url_content(url: str) -> str | None:
    """
    Extract the main content from a URL using Zyte API.
    
    Args:
        url: The URL to extract content from
        
    Returns:
        The extracted article text content
        
    Raises:
        Exception: If there's an error downloading or parsing the article
    """
    if not ZYTE_API_KEY:
        raise Exception("ZYTE_API_KEY environment variable is not set")
        
    try:
        client = ZyteAPI(api_key=ZYTE_API_KEY)
        zyte_query = {
            "url": url,
            "article": True,
            "articleOptions": {"extractFrom":"browserHtml"},
        }
        response = client.get(zyte_query)
        
        status_code = response.get("statusCode", 0)
        if status_code != 200:
            logging.warning(f"Failed to extract content from URL {url}: {status_code}")
            return None
        
        article = response.get("article", {})
        if not article:
            logging.warning(f"No article found in response for URL {url}")
            return None
        
        article_body = article.get("articleBody", "")
        if not article_body:
            logging.warning(f"No article body found in response for URL {url}")
            return None
        
        return article_body
    except Exception as e:
        logging.warning(f"Failed to extract content from URL {url}: {str(e)}")
        return None


def get_rag_sources(query: str, *, num_sources: int = 10) -> list[RAGSource]:
    """
    Get a list of RAG sources for a given query using Google search.
    
    Args:
        query: The search query string
        num_sources: Number of sources to return (default: 10)
        
    Returns:
        List of RAG sources, where each source is a dictionary containing
        keys like 'url', 'title', 'content', etc.
    """
    logging.info(f" Getting RAG sources for query: `{query}`")
    search_results = google_search(query, num_results=num_sources)
    sources = []
    for result in search_results:
        url = result.get("link", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")

        if url.endswith(".pdf"):
            # Zyte won't extract content from PDFs, so we skip them
            continue

        content = get_url_content(url)
        if content:
            logging.info(f" -> Found source: `{url}`")
            sources.append(RAGSource(url=url, title=title, content=content, relevant_snippet=snippet))
        else:
            logging.info(f" -> Skipping source: `{url}`")
    return sources


def build_rag_query(entity_name: str, props: Properties) -> str:
    """Build a query for RAG using the entity name."""
    value = get_value(props)
    return f"{value} of {entity_name}"
    

def build_rag_extraction_prompt(query: str, source: RAGSource) -> str:
    """Build a prompt for extracting values from a single source."""
    return f"""Given the following query and source, extract the value that answers the query. If a value is found, provide only the value with no additional text or formatting. If no clear value can be found, respond with "UNKNOWN".

Query: `{query}`

Source title: `{source.title}`
Source URL: `{source.url}`
Source snippet relevant to the query: `{source.relevant_snippet}`
Source full content: `{source.content}`"""


def extract_rag_values_from_sources(
    query: str,
    sources: list[RAGSource],
    model_id: str = "gpt-4o-2024-11-20",
    temperature: float = 0.7,
    max_tokens: int = 100,
) -> list[RAGValue]:
    """
    Extract a range of values for a given query from a list of sources using an OpenAI model.
    Processes each source individually and returns all non-UNKNOWN values found.
    """
    extracted_values: list[RAGValue] = []
    
    logging.info(f"### Query: `{query}`")
    
    for source in sources:
        prompt = build_rag_extraction_prompt(query, source)
        
        logging.info(f"### Extracting value from source: `{source.url}`")
        logging.info(f"Prompt: `{prompt}`")

        response = generate_oa_response_sync(
            prompt=prompt,
            model_id=model_id,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        
        logging.info(f" -> Extracted value: `{response}`")
        logging.info("-" * 80)
        
        if response and response.strip().lower() != "unknown":
            extracted_values.append(RAGValue(value=response.strip(), source=source))
    
    return extracted_values

