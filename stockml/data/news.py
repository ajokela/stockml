"""News API integration for fetching stock-related news"""

from datetime import datetime, timedelta
from typing import List, Optional
import os


class NewsClient:
    """Client for fetching news articles about stocks"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize NewsClient

        Args:
            api_key: NewsAPI.org API key. If not provided, looks for
                     NEWS_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazily initialize the NewsAPI client"""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "NewsAPI key required. Set NEWS_API_KEY environment variable "
                    "or pass api_key to NewsClient constructor."
                )
            from newsapi import NewsApiClient
            self._client = NewsApiClient(api_key=self.api_key)
        return self._client

    def fetch_news(
        self,
        query: str,
        days: int = 7,
        max_articles: int = 20,
        language: str = "en"
    ) -> List[dict]:
        """Fetch news articles matching a query

        Args:
            query: Search query (e.g., ticker symbol or company name)
            days: Number of days back to search
            max_articles: Maximum number of articles to return
            language: Language code (default: "en")

        Returns:
            List of article dicts with keys:
            - title: Article title
            - description: Article description/summary
            - content: Article content (may be truncated)
            - source: Source name
            - url: Article URL
            - published_at: Publication datetime
        """
        client = self._get_client()

        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        response = client.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language=language,
            sort_by="relevancy",
            page_size=max_articles,
        )

        articles = []
        for article in response.get("articles", []):
            articles.append({
                "title": article.get("title"),
                "description": article.get("description"),
                "content": article.get("content"),
                "source": article.get("source", {}).get("name"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "author": article.get("author"),
            })

        return articles

    def fetch_stock_news(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        days: int = 7,
        max_articles: int = 20
    ) -> List[dict]:
        """Fetch news specifically about a stock

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            company_name: Company name for broader search (e.g., "Apple")
            days: Number of days back to search
            max_articles: Maximum articles to return

        Returns:
            List of article dicts
        """
        # Build query with ticker and optionally company name
        query_parts = [ticker]
        if company_name:
            query_parts.append(company_name)

        query = " OR ".join(query_parts)
        return self.fetch_news(query, days=days, max_articles=max_articles)

    def is_configured(self) -> bool:
        """Check if the client has an API key configured"""
        return self.api_key is not None


class MockNewsClient:
    """Mock news client for testing without API key"""

    def __init__(self):
        self.api_key = None

    def fetch_news(self, query: str, **kwargs) -> List[dict]:
        """Return empty list for mock client"""
        return []

    def fetch_stock_news(self, ticker: str, **kwargs) -> List[dict]:
        """Return empty list for mock client"""
        return []

    def is_configured(self) -> bool:
        return False
