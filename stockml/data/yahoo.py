"""Yahoo Finance data fetcher"""

from typing import Optional
import yfinance as yf
import pandas as pd


class YahooFinanceClient:
    """Client for fetching stock data from Yahoo Finance"""

    def __init__(self):
        self._cache = {}

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create a cached ticker object"""
        if symbol not in self._cache:
            self._cache[symbol] = yf.Ticker(symbol)
        return self._cache[symbol]

    def get_stock_info(self, symbol: str) -> dict:
        """Get basic stock information

        Returns dict with keys like: shortName, sector, industry,
        marketCap, currentPrice, etc.
        """
        ticker = self._get_ticker(symbol)
        return ticker.info

    def get_price_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical OHLCV data

        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with Open, High, Low, Close, Volume columns
        """
        ticker = self._get_ticker(symbol)
        return ticker.history(period=period, interval=interval)

    def get_dividends(self, symbol: str) -> pd.Series:
        """Get dividend history

        Returns:
            Series with dividend amounts indexed by date
        """
        ticker = self._get_ticker(symbol)
        return ticker.dividends

    def get_financials(self, symbol: str) -> dict:
        """Get financial statements data

        Returns:
            Dict containing income_stmt, balance_sheet, cash_flow
        """
        ticker = self._get_ticker(symbol)
        return {
            "income_stmt": ticker.income_stmt,
            "balance_sheet": ticker.balance_sheet,
            "cash_flow": ticker.cash_flow,
            "quarterly_income_stmt": ticker.quarterly_income_stmt,
            "quarterly_balance_sheet": ticker.quarterly_balance_sheet,
            "quarterly_cash_flow": ticker.quarterly_cash_flow,
        }

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price"""
        info = self.get_stock_info(symbol)
        return info.get("currentPrice") or info.get("regularMarketPrice")

    def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """Get analyst recommendations"""
        ticker = self._get_ticker(symbol)
        return ticker.recommendations

    def clear_cache(self):
        """Clear the ticker cache"""
        self._cache.clear()
