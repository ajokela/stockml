"""Alpha Vantage API client"""

import os
from typing import Optional, List
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import json


class AlphaVantageClient:
    """Client for fetching data from Alpha Vantage API"""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Alpha Vantage client

        Args:
            api_key: Alpha Vantage API key. If not provided, looks for
                     ALPHAVANTAGE_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")

    def _request(self, params: dict) -> dict:
        """Make API request"""
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key required. Set ALPHAVANTAGE_API_KEY "
                "environment variable or pass api_key to constructor."
            )

        params["apikey"] = self.api_key
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.BASE_URL}?{query_string}"

        try:
            response = urlopen(url)
            data = json.loads(response.read().decode("utf-8"))

            # Check for API errors
            if "Error Message" in data:
                raise Exception(f"Alpha Vantage error: {data['Error Message']}")
            if "Note" in data:
                # Rate limit message
                raise Exception(f"Alpha Vantage rate limit: {data['Note']}")

            return data
        except HTTPError as e:
            raise Exception(f"Alpha Vantage API error: {e.code} {e.reason}")
        except URLError as e:
            raise Exception(f"Network error: {e.reason}")

    def is_configured(self) -> bool:
        """Check if the client has an API key configured"""
        return self.api_key is not None

    # ============ Company Overview ============

    def get_company_overview(self, symbol: str) -> dict:
        """Get comprehensive company fundamentals

        Returns: Market cap, P/E, PEG, book value, dividend yield,
                 EPS, revenue, profit margin, 52-week highs/lows, etc.
        """
        return self._request({
            "function": "OVERVIEW",
            "symbol": symbol
        })

    # ============ News Sentiment ============

    def get_news_sentiment(
        self,
        tickers: str,
        topics: Optional[str] = None,
        time_from: Optional[str] = None,
        limit: int = 50
    ) -> dict:
        """Get news articles with sentiment scores

        Args:
            tickers: Comma-separated ticker symbols (e.g., "AAPL,MSFT")
            topics: Filter by topics (e.g., "technology", "earnings")
            time_from: Start time in YYYYMMDDTHHMM format
            limit: Number of articles (max 1000)

        Returns:
            Dict with feed of articles, each containing:
            - title, summary, source, url
            - overall_sentiment_score (-1 to 1)
            - overall_sentiment_label (Bearish, Neutral, Bullish)
            - ticker_sentiment with relevance_score
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": tickers,
            "limit": limit,
            "sort": "RELEVANCE"
        }
        if topics:
            params["topics"] = topics
        if time_from:
            params["time_from"] = time_from

        return self._request(params)

    # ============ Earnings ============

    def get_earnings(self, symbol: str) -> dict:
        """Get quarterly and annual earnings data

        Returns:
            Dict with annualEarnings and quarterlyEarnings lists
        """
        return self._request({
            "function": "EARNINGS",
            "symbol": symbol
        })

    # ============ Technical Indicators ============

    def get_rsi(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close"
    ) -> dict:
        """Get Relative Strength Index"""
        return self._request({
            "function": "RSI",
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type
        })

    def get_macd(
        self,
        symbol: str,
        interval: str = "daily",
        series_type: str = "close"
    ) -> dict:
        """Get MACD indicator"""
        return self._request({
            "function": "MACD",
            "symbol": symbol,
            "interval": interval,
            "series_type": series_type
        })

    def get_sma(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 20,
        series_type: str = "close"
    ) -> dict:
        """Get Simple Moving Average"""
        return self._request({
            "function": "SMA",
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type
        })

    def get_ema(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 20,
        series_type: str = "close"
    ) -> dict:
        """Get Exponential Moving Average"""
        return self._request({
            "function": "EMA",
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type
        })

    def get_bbands(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 20,
        series_type: str = "close"
    ) -> dict:
        """Get Bollinger Bands"""
        return self._request({
            "function": "BBANDS",
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type
        })

    # ============ Economic Indicators ============

    def get_real_gdp(self, interval: str = "annual") -> dict:
        """Get US Real GDP data"""
        return self._request({
            "function": "REAL_GDP",
            "interval": interval
        })

    def get_federal_funds_rate(self) -> dict:
        """Get Federal Funds Rate"""
        return self._request({
            "function": "FEDERAL_FUNDS_RATE"
        })

    def get_cpi(self, interval: str = "monthly") -> dict:
        """Get Consumer Price Index (inflation)"""
        return self._request({
            "function": "CPI",
            "interval": interval
        })

    def get_unemployment(self) -> dict:
        """Get US unemployment rate"""
        return self._request({
            "function": "UNEMPLOYMENT"
        })


class MockAlphaVantageClient:
    """Mock Alpha Vantage client for when no API key is available"""

    def __init__(self):
        self.api_key = None

    def is_configured(self) -> bool:
        return False

    def get_company_overview(self, symbol: str) -> dict:
        return {}

    def get_news_sentiment(self, tickers: str, **kwargs) -> dict:
        return {}

    def get_earnings(self, symbol: str) -> dict:
        return {}

    def get_rsi(self, symbol: str, **kwargs) -> dict:
        return {}

    def get_macd(self, symbol: str, **kwargs) -> dict:
        return {}

    def get_sma(self, symbol: str, **kwargs) -> dict:
        return {}

    def get_ema(self, symbol: str, **kwargs) -> dict:
        return {}

    def get_bbands(self, symbol: str, **kwargs) -> dict:
        return {}
