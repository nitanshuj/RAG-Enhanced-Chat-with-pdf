"""
Web search module for hybrid knowledge retrieval.

Implements DuckDuckGo search for augmenting document-based RAG with web results.
"""

from typing import List, Dict, Any, Optional
from src.logger import get_logger

logger = get_logger(__name__)

# Optional imports for web search
try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    logger.warning("duckduckgo-search not available. Web search disabled.")


class WebSearcher:
    """
    Web search integration for hybrid knowledge retrieval.

    Features:
    - DuckDuckGo search (free, no API key required)
    - Configurable number of results
    - Result filtering and formatting
    - Safe fallback if search unavailable
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize web searcher.

        Args:
            enabled: Enable web search functionality
        """
        self.enabled = enabled and WEB_SEARCH_AVAILABLE
        self.ddgs = None

        if self.enabled:
            try:
                self.ddgs = DDGS()
                logger.info("WebSearcher initialized with DuckDuckGo")
            except Exception as e:
                logger.warning(f"Failed to initialize DuckDuckGo search: {e}")
                self.enabled = False

    def search(self, query: str, max_results: int = 2,
               region: str = "wt-wt", safesearch: str = "moderate") -> List[Dict[str, Any]]:
        """
        Perform web search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 2 for 15% weight)
            region: Search region (default: "wt-wt" for worldwide)
            safesearch: Safe search setting ("on", "moderate", "off")

        Returns:
            List of search result dictionaries with 'title', 'body', 'href'
        """
        if not self.enabled or not self.ddgs:
            logger.warning("Web search not available")
            return []

        try:
            # Perform search
            results = self.ddgs.text(
                keywords=query,
                region=region,
                safesearch=safesearch,
                max_results=max_results
            )

            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "text": f"{result.get('title', '')}\n\n{result.get('body', '')}",
                    "title": result.get('title', 'No Title'),
                    "url": result.get('href', ''),
                    "snippet": result.get('body', ''),
                    "source": "web",
                    "rank": i + 1,
                    "metadata": {
                        "source": "web",
                        "url": result.get('href', ''),
                        "search_engine": "DuckDuckGo"
                    }
                })

            logger.info(f"Web search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []

    def should_use_web_search(self, doc_results: List[Dict[str, Any]],
                             query: str,
                             min_confidence: float = 0.6,
                             explicit_request: bool = False) -> bool:
        """
        Determine if web search should be triggered.

        Args:
            doc_results: Results from document search
            query: User query
            min_confidence: Minimum confidence threshold
            explicit_request: User explicitly requested web search

        Returns:
            Boolean indicating whether to use web search
        """
        if not self.enabled:
            return False

        # Always use if explicitly requested
        if explicit_request:
            return True

        # Check if document results have low confidence
        if not doc_results:
            return True

        # Check average confidence of document results
        total_confidence = sum(
            doc.get("metadata", {}).get("confidence", 1.0)
            for doc in doc_results
        )
        avg_confidence = total_confidence / len(doc_results)

        if avg_confidence < min_confidence:
            logger.info(f"Low document confidence ({avg_confidence:.2f}), triggering web search")
            return True

        # Check for time-sensitive keywords
        time_sensitive_keywords = [
            "latest", "recent", "current", "today", "now", "2024", "2025",
            "news", "update", "breaking", "trend"
        ]
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in time_sensitive_keywords):
            logger.info("Time-sensitive query detected, triggering web search")
            return True

        return False

    def combine_results(self, doc_results: List[Dict[str, Any]],
                       web_results: List[Dict[str, Any]],
                       doc_weight: float = 0.85) -> List[Dict[str, Any]]:
        """
        Combine document and web results with weighting.

        Args:
            doc_results: Results from document search
            web_results: Results from web search
            doc_weight: Weight for document results (default: 0.85 for 85%)

        Returns:
            Combined and weighted results list
        """
        web_weight = 1.0 - doc_weight

        # Calculate how many results from each source
        total_results = len(doc_results) + len(web_results)
        doc_count = int(total_results * doc_weight)
        web_count = total_results - doc_count

        # Take top results from each source
        combined = doc_results[:doc_count] + web_results[:web_count]

        logger.info(f"Combined results: {doc_count} from documents, {web_count} from web")
        return combined

    def format_web_context(self, web_results: List[Dict[str, Any]]) -> str:
        """
        Format web results for LLM context.

        Args:
            web_results: List of web search results

        Returns:
            Formatted string for LLM context
        """
        if not web_results:
            return ""

        context_parts = ["\n\n--- Web Search Results ---\n"]

        for i, result in enumerate(web_results, 1):
            context_parts.append(
                f"Web Result {i} (Source: {result.get('url', 'Unknown')}):\n"
                f"{result.get('text', '')}\n"
            )

        return "\n".join(context_parts)


# Singleton instance
_web_searcher_instance: Optional[WebSearcher] = None


def get_web_searcher(enabled: bool = True) -> WebSearcher:
    """
    Get or create singleton web searcher instance.

    Args:
        enabled: Enable web search functionality

    Returns:
        WebSearcher instance
    """
    global _web_searcher_instance
    if _web_searcher_instance is None:
        _web_searcher_instance = WebSearcher(enabled=enabled)
    return _web_searcher_instance
