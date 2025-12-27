"""Financial Modeling Prep API client"""

import os
from typing import Optional, List
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import json


class FMPClient:
    """Client for fetching data from Financial Modeling Prep API"""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize FMP client

        Args:
            api_key: FMP API key. If not provided, looks for
                     FMP_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY")

    def _request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make API request"""
        if not self.api_key:
            raise ValueError(
                "FMP API key required. Set FMP_API_KEY environment variable "
                "or pass api_key to FMPClient constructor."
            )

        url = f"{self.BASE_URL}/{endpoint}?apikey={self.api_key}"

        if params:
            for key, value in params.items():
                url += f"&{key}={value}"

        try:
            response = urlopen(url)
            data = json.loads(response.read().decode("utf-8"))
            return data
        except HTTPError as e:
            raise Exception(f"FMP API error: {e.code} {e.reason}")
        except URLError as e:
            raise Exception(f"Network error: {e.reason}")

    def is_configured(self) -> bool:
        """Check if the client has an API key configured"""
        return self.api_key is not None

    # ============ Company Profile ============

    def get_company_profile(self, symbol: str) -> dict:
        """Get detailed company profile

        Returns: Company info including sector, industry, description,
                 CEO, employees, market cap, etc.
        """
        data = self._request(f"profile/{symbol}")
        return data[0] if data else {}

    # ============ Financial Statements ============

    def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> List[dict]:
        """Get income statements

        Args:
            symbol: Stock ticker
            period: "annual" or "quarter"
            limit: Number of periods to fetch
        """
        return self._request(
            f"income-statement/{symbol}",
            {"period": period, "limit": limit}
        )

    def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> List[dict]:
        """Get balance sheets"""
        return self._request(
            f"balance-sheet-statement/{symbol}",
            {"period": period, "limit": limit}
        )

    def get_cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> List[dict]:
        """Get cash flow statements"""
        return self._request(
            f"cash-flow-statement/{symbol}",
            {"period": period, "limit": limit}
        )

    def get_financial_ratios(self, symbol: str, limit: int = 5) -> List[dict]:
        """Get pre-calculated financial ratios

        Includes: P/E, P/B, ROE, ROA, current ratio, debt ratios, etc.
        """
        return self._request(f"ratios/{symbol}", {"limit": limit})

    def get_key_metrics(self, symbol: str, limit: int = 5) -> List[dict]:
        """Get key financial metrics

        Includes: Revenue per share, FCF per share, book value, etc.
        """
        return self._request(f"key-metrics/{symbol}", {"limit": limit})

    # ============ Valuation ============

    def get_dcf(self, symbol: str) -> dict:
        """Get discounted cash flow (DCF) valuation

        Returns estimated fair value based on DCF model
        """
        data = self._request(f"discounted-cash-flow/{symbol}")
        return data[0] if data else {}

    def get_rating(self, symbol: str) -> dict:
        """Get FMP's proprietary stock rating

        Returns overall rating and component scores
        """
        data = self._request(f"rating/{symbol}")
        return data[0] if data else {}

    def get_enterprise_value(self, symbol: str, limit: int = 5) -> List[dict]:
        """Get enterprise value metrics"""
        return self._request(f"enterprise-values/{symbol}", {"limit": limit})

    # ============ Analyst Data ============

    def get_analyst_estimates(self, symbol: str, limit: int = 4) -> List[dict]:
        """Get analyst earnings estimates

        Returns EPS and revenue estimates for upcoming quarters
        """
        return self._request(f"analyst-estimates/{symbol}", {"limit": limit})

    def get_price_target(self, symbol: str) -> List[dict]:
        """Get analyst price targets

        Returns individual analyst price targets and ratings
        """
        return self._request(f"price-target/{symbol}")

    def get_price_target_summary(self, symbol: str) -> dict:
        """Get price target consensus summary

        Returns average, high, low price targets
        """
        data = self._request(f"price-target-summary/{symbol}")
        return data[0] if data else {}

    def get_price_target_consensus(self, symbol: str) -> dict:
        """Get consensus price target"""
        data = self._request(f"price-target-consensus/{symbol}")
        return data[0] if data else {}

    def get_analyst_recommendations(self, symbol: str) -> List[dict]:
        """Get analyst buy/sell/hold recommendations over time"""
        return self._request(f"analyst-stock-recommendations/{symbol}")

    def get_grades(self, symbol: str, limit: int = 10) -> List[dict]:
        """Get analyst grade changes (upgrades/downgrades)"""
        return self._request(f"grade/{symbol}", {"limit": limit})

    # ============ Insider & Institutional ============

    def get_insider_trading(self, symbol: str, limit: int = 20) -> List[dict]:
        """Get insider trading activity

        Returns executive buy/sell transactions
        """
        return self._request(f"insider-trading", {"symbol": symbol, "limit": limit})

    def get_institutional_holders(self, symbol: str) -> List[dict]:
        """Get institutional holders (13F data)"""
        return self._request(f"institutional-holder/{symbol}")

    # ============ News & Events ============

    def get_stock_news(self, symbol: str, limit: int = 20) -> List[dict]:
        """Get stock-specific news articles"""
        return self._request(f"stock_news", {"tickers": symbol, "limit": limit})

    def get_press_releases(self, symbol: str, limit: int = 10) -> List[dict]:
        """Get company press releases"""
        return self._request(f"press-releases/{symbol}", {"limit": limit})

    def get_earnings_calendar(
        self,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[dict]:
        """Get earnings calendar

        Args:
            symbol: Filter by ticker (optional)
            from_date: Start date YYYY-MM-DD
            to_date: End date YYYY-MM-DD
        """
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        if symbol:
            return self._request(f"historical/earning_calendar/{symbol}", params)
        return self._request("earning_calendar", params)

    def get_earnings_transcripts(
        self,
        symbol: str,
        year: Optional[int] = None,
        quarter: Optional[int] = None
    ) -> List[dict]:
        """Get earnings call transcripts

        Args:
            symbol: Stock ticker
            year: Filter by year (optional)
            quarter: Filter by quarter 1-4 (optional)

        Returns:
            List of transcripts with: date, quarter, year, content (full text)
        """
        params = {}
        if year:
            params["year"] = year
        if quarter:
            params["quarter"] = quarter

        return self._request(f"earning_call_transcript/{symbol}", params)

    def get_latest_transcript(self, symbol: str) -> dict:
        """Get the most recent earnings call transcript

        Args:
            symbol: Stock ticker

        Returns:
            Most recent transcript dict or empty dict if none found
        """
        transcripts = self.get_earnings_transcripts(symbol)
        if transcripts and len(transcripts) > 0:
            # Transcripts are returned newest first
            return transcripts[0]
        return {}

    # ============ Quote & Price ============

    def get_quote(self, symbol: str) -> dict:
        """Get real-time quote"""
        data = self._request(f"quote/{symbol}")
        return data[0] if data else {}

    def get_quote_short(self, symbol: str) -> dict:
        """Get simplified quote (price, volume, change)"""
        data = self._request(f"quote-short/{symbol}")
        return data[0] if data else {}

    # ============ Dividends ============

    def get_dividend_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[dict]:
        """Get upcoming dividend calendar"""
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._request("stock_dividend_calendar", params)

    def get_historical_dividends(self, symbol: str) -> List[dict]:
        """Get historical dividend data"""
        return self._request(f"historical-price-full/stock_dividend/{symbol}")

    # ============ Stock Peers ============

    def get_stock_peers(self, symbol: str) -> List[str]:
        """Get list of peer/competitor stock symbols

        Uses FMP's stock peers endpoint to find companies in the same
        sector with similar market cap.

        Args:
            symbol: Stock ticker

        Returns:
            List of peer stock symbols
        """
        if not self.api_key:
            return []

        # Stock peers uses v4 API
        url = f"https://financialmodelingprep.com/api/v4/stock_peers?symbol={symbol}&apikey={self.api_key}"

        try:
            response = urlopen(url)
            data = json.loads(response.read().decode("utf-8"))
            if data and len(data) > 0:
                return data[0].get("peersList", [])
            return []
        except (HTTPError, URLError):
            return []

    # ============ Economic Data ============

    def get_treasury_rates(self) -> List[dict]:
        """Get current treasury rates"""
        return self._request("treasury")

    def get_economic_indicators(self, name: str) -> List[dict]:
        """Get economic indicator data

        Args:
            name: Indicator name (e.g., "GDP", "inflation", "unemployment")
        """
        return self._request(f"economic", {"name": name})


class MockFMPClient:
    """Mock FMP client for when no API key is available"""

    def __init__(self):
        self.api_key = None

    def is_configured(self) -> bool:
        return False

    def get_company_profile(self, symbol: str) -> dict:
        return {}

    def get_dcf(self, symbol: str) -> dict:
        return {}

    def get_rating(self, symbol: str) -> dict:
        return {}

    def get_analyst_estimates(self, symbol: str, limit: int = 4) -> List[dict]:
        return []

    def get_price_target_summary(self, symbol: str) -> dict:
        return {}

    def get_price_target_consensus(self, symbol: str) -> dict:
        return {}

    def get_insider_trading(self, symbol: str, limit: int = 20) -> List[dict]:
        return []

    def get_stock_news(self, symbol: str, limit: int = 20) -> List[dict]:
        return []

    def get_grades(self, symbol: str, limit: int = 10) -> List[dict]:
        return []

    def get_financial_ratios(self, symbol: str, limit: int = 5) -> List[dict]:
        return []

    def get_key_metrics(self, symbol: str, limit: int = 5) -> List[dict]:
        return []

    def get_earnings_calendar(self, symbol: str = None, **kwargs) -> List[dict]:
        return []

    def get_earnings_transcripts(self, symbol: str, **kwargs) -> List[dict]:
        return []

    def get_latest_transcript(self, symbol: str) -> dict:
        return {}

    def get_stock_peers(self, symbol: str) -> List[str]:
        return []

    def get_quote(self, symbol: str) -> dict:
        return {}
