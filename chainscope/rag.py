import logging
import os
from typing import Any

from bs4 import BeautifulSoup
from google.oauth2 import service_account
from googleapiclient.discovery import build
from zyte_api import ZyteAPI

from chainscope.api_utils.open_ai_utils import \
    generate_oa_web_search_response_sync
from chainscope.properties import get_value
from chainscope.typing import *

GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
ZYTE_API_KEY = os.getenv("ZYTE_API_KEY")


def google_search(query: str, *, num_results: int = 10) -> list[dict[str, Any]]:
    """
    Perform a Google search using the Custom Search JSON API with service account authentication.
    
    Args:
        query: The search query string
        num_results: Number of results to return (max 10 per request)
        
    Returns:
        List of search results, where each result is a dictionary containing
        keys like 'title', 'link', 'snippet', etc.
    """
    assert GOOGLE_SERVICE_ACCOUNT_FILE is not None or GOOGLE_SEARCH_API_KEY is not None, "GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SEARCH_API_KEY environment variable must be set"
    try:
        if GOOGLE_SERVICE_ACCOUNT_FILE:
            # Create credentials using service account file
            credentials = service_account.Credentials.from_service_account_file(
                GOOGLE_SERVICE_ACCOUNT_FILE,
                scopes=['https://www.googleapis.com/auth/cse']
            )
            
            # Build the service object with service account credentials
            service = build('customsearch', 'v1', credentials=credentials)
        else:
            # Build the service object with API key
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


def get_url_content(url: str) -> tuple[str, str] | None:
    """
    Extract the main content from a URL using Zyte API.
    
    Args:
        url: The URL to extract content from
        
    Returns:
        A tuple containing the extracted article text content and the HTML content
        
    Raises:
        Exception: If there's an error downloading or parsing the article
    """
    if not ZYTE_API_KEY:
        raise Exception("ZYTE_API_KEY environment variable is not set")
        
    try:
        client = ZyteAPI(api_key=ZYTE_API_KEY)
        zyte_query = {
            "url": url,
            "browserHtml": True,
            "article": True,
            "articleOptions": {"extractFrom":"browserHtml"},
        }
        response = client.get(zyte_query)
        
        status_code = response.get("statusCode", 0)
        if status_code != 200:
            logging.warning(f"Failed to extract content from URL {url}: {status_code}")
            return None
        
        html = response.get("browserHtml", "")
        if not html:
            logging.warning(f"No HTML found in response for URL {url}")
            return None
        
        article = response.get("article", {})
        if not article:
            logging.warning(f"No article found in response for URL {url}")
            return None
        
        article_body = article.get("articleBody", "")
        if not article_body:
            logging.warning(f"No article body found in response for URL {url}")
            return None
        
        return article_body, html
    except Exception as e:
        logging.warning(f"Failed to extract content from URL {url}: {str(e)}")
        return None
    

def extract_wikipedia_infobox_data(html_content: str) -> dict[str, dict[str, str]]:
    """
    Parses the first 'infobox' table and returns a dict of dicts.
    The top-level keys are 'section headers' (like "Physical properties").
    Each section maps to its own dict of property->value pairs.
    Rows that do not belong to a section with a header
    are placed into a 'Miscellaneous' or similar default group.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    infobox = soup.find("table", class_="infobox")
    if not infobox:
        return {}
    
    # This will map: header_text -> { property_key: property_value, ... }
    infobox_data = {}
    current_header = None
    
    # We'll iterate over every row in the infobox (allowing recursion to find <tbody> child rows).
    for row in infobox.find_all("tr"):
        # First see if this row is a "section header" row: <th colspan="2" class="infobox-header">Some Header</th>
        header_cell = row.find("th", class_="infobox-header")
        if header_cell:
            # Grab the header text
            current_header = header_cell.get_text(strip=True)
            # Initialize this header in our dict if not present
            infobox_data.setdefault(current_header, {})
            continue
        
        # If no "infobox-header", check if it's a property row (1 <th> + 1 <td>)
        prop_th = row.find("th", recursive=False)
        prop_td = row.find("td", recursive=False)
        
        # If we don't see a property row, skip it
        if not prop_th or not prop_td:
            continue
        
        # If there's a nested table, skip (complex data)
        if prop_td.find("table"):
            continue
        
        # If no header has been encountered yet, put this under a default group
        if not current_header:
            current_header = "Miscellaneous"
            infobox_data.setdefault(current_header, {})
        
        # Extract clean text for key and value
        key = prop_th.get_text(strip=True)
        value = prop_td.get_text(" ", strip=True)
        
        # Store in our dict
        infobox_data[current_header][key] = value
    
    return infobox_data


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

        skip_extensions = [".pdf", ".docx", ".doc", ".xls", ".xlsx", ".ppt", ".pptx", ".csv"]
        if any(url.endswith(ext) for ext in skip_extensions):
            # Zyte won't extract content from PDFs or other non-HTML content, so we skip them
            continue

        content_extraction_result = get_url_content(url)
        if content_extraction_result:
            content, html = content_extraction_result
            logging.info(f" -> Found source: `{url}`")

            if "wikipedia.org" in url:
                # Wikipedia pages have a lot of content in their infoboxes, which is not parsed by Zyte, so we need to extract that manually
                infobox_data = extract_wikipedia_infobox_data(html)
                infobox_content = ""
                for header, properties in infobox_data.items():
                    infobox_content += f"# {header}\n"
                    for key, value in properties.items():
                        infobox_content += f"- {key}: {value}\n"
            
                content = f"Wikipedia infobox:\n{infobox_content}\n\n{content}"

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
    prompt = f"""Given the following query and source, extract the value that answers the query. If a value is found, provide only the value with no additional text or formatting. If no clear value can be found, respond with "UNKNOWN".

Query: `{query}`

Source title: `{source.title}`
Source URL: `{source.url}`"""

    if source.relevant_snippet:
        prompt += f"\nSource snippet relevant to the query: `{source.relevant_snippet}`"
    if source.content:
        prompt += f"\nSource full content: `{source.content}`"

    return prompt


def get_openai_web_search_rag_values(
    query: str,
    model_id: Literal["gpt-4o-search-preview", "gpt-4o-mini-search-preview"],
    max_new_tokens: int = 1000,
    search_context_size: Literal["low", "medium", "high"] = "medium",
    user_location_country: str | None = "US",
    user_location_city: str | None = "San Francisco",
    user_location_region: str | None = "California",
) -> list[RAGValue]:
    """Get RAG values using OpenAI's web search API."""
    prompt_template = """Given the following query, search the web and extract the value that answers the query from each source. Provide only the value found in each source with the appropriate inline citation (all in english). If a source does not have a clear value for the query, omit it from the answer. You should follow the format:

- **Source 1**: value 1
- **Source 2**: value 2
Etc.

Query: `{query}`"""

    prompt = prompt_template.format(query=query)
    response, annotations = generate_oa_web_search_response_sync(
        prompt=prompt,
        model_id=model_id,
        user_location_country=user_location_country,
        user_location_city=user_location_city,
        user_location_region=user_location_region,
        max_new_tokens=max_new_tokens,
        search_context_size=search_context_size,
    )
    
    values = []
    for annotation in annotations:
        citation = annotation.url_citation
        clean_url = citation.url.replace("?utm_source=openai", "")
        source = RAGSource(url=clean_url, title=citation.title, content=None, relevant_snippet=None)
        
        # Get the value between the first colon to the right of the start of the annotation
        # E.g., the following string: 
        # "- **Source 1**: 28,788 ([illinois-demographics.com](https://www.illinois-demographics.com/60459-demographics?utm_source=openai))"
        # Will have an annotation: 
        # Annotation(type='url_citation', url_citation=AnnotationURLCitation(end_index=128, start_index=23, title='60459 Demographics | Current Illinois Census Data', url='https://www.illinois-demographics.com/60459-demographics?utm_source=openai'))
        relevant_text = response[:citation.start_index]
        source_end_str = "**:"
        if source_end_str not in relevant_text:
            continue
        val = relevant_text.split(source_end_str)[-1].strip()

        # remove any parentheses from the value
        val = val.split("(")[0].strip()

        if val == "":
            continue
        values.append(RAGValue(value=val, source=source))

    return values