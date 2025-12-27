"""Technical analysis indicators and signals"""

from typing import Tuple
import pandas as pd
import numpy as np


class TechnicalAnalyzer:
    """Calculate technical indicators and generate trading signals"""

    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index

        RSI ranges from 0-100:
        - Below 30: Oversold (potential buy signal)
        - Above 70: Overbought (potential sell signal)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range (volatility indicator)"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    def calculate_support_resistance(
        prices: pd.Series,
        window: int = 20
    ) -> Tuple[float, float]:
        """Calculate simple support and resistance levels

        Uses rolling min/max as basic support/resistance
        """
        recent = prices.tail(window)
        support = recent.min()
        resistance = recent.max()
        return support, resistance

    def analyze(self, history: pd.DataFrame) -> dict:
        """Perform full technical analysis on price history

        Args:
            history: DataFrame with Open, High, Low, Close, Volume columns

        Returns:
            Dict containing all technical indicators and signals
        """
        close = history["Close"]
        high = history["High"]
        low = history["Low"]
        volume = history["Volume"]

        # Calculate indicators
        sma_20 = self.calculate_sma(close, 20)
        sma_50 = self.calculate_sma(close, 50)
        sma_200 = self.calculate_sma(close, 200)
        ema_12 = self.calculate_ema(close, 12)
        ema_26 = self.calculate_ema(close, 26)

        rsi = self.calculate_rsi(close)
        macd_line, signal_line, macd_hist = self.calculate_macd(close)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
        atr = self.calculate_atr(high, low, close)

        support, resistance = self.calculate_support_resistance(close)

        # Get latest values
        current_price = close.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        latest_atr = atr.iloc[-1]

        # Generate signals
        signals = []

        # RSI signals
        if latest_rsi < 30:
            signals.append({"type": "rsi", "signal": "oversold", "strength": "strong_buy"})
        elif latest_rsi < 40:
            signals.append({"type": "rsi", "signal": "approaching_oversold", "strength": "buy"})
        elif latest_rsi > 70:
            signals.append({"type": "rsi", "signal": "overbought", "strength": "strong_sell"})
        elif latest_rsi > 60:
            signals.append({"type": "rsi", "signal": "approaching_overbought", "strength": "sell"})

        # MACD signals
        if latest_macd > latest_signal and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            signals.append({"type": "macd", "signal": "bullish_crossover", "strength": "buy"})
        elif latest_macd < latest_signal and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            signals.append({"type": "macd", "signal": "bearish_crossover", "strength": "sell"})

        # Moving average signals
        if current_price > sma_50.iloc[-1] > sma_200.iloc[-1]:
            signals.append({"type": "trend", "signal": "bullish_trend", "strength": "buy"})
        elif current_price < sma_50.iloc[-1] < sma_200.iloc[-1]:
            signals.append({"type": "trend", "signal": "bearish_trend", "strength": "sell"})

        # Golden/Death cross
        if sma_50.iloc[-1] > sma_200.iloc[-1] and sma_50.iloc[-2] <= sma_200.iloc[-2]:
            signals.append({"type": "cross", "signal": "golden_cross", "strength": "strong_buy"})
        elif sma_50.iloc[-1] < sma_200.iloc[-1] and sma_50.iloc[-2] >= sma_200.iloc[-2]:
            signals.append({"type": "cross", "signal": "death_cross", "strength": "strong_sell"})

        # Bollinger Band signals
        if current_price < bb_lower.iloc[-1]:
            signals.append({"type": "bollinger", "signal": "below_lower_band", "strength": "buy"})
        elif current_price > bb_upper.iloc[-1]:
            signals.append({"type": "bollinger", "signal": "above_upper_band", "strength": "sell"})

        # Calculate overall trend
        trend = "neutral"
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend = "bullish"
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            trend = "bearish"

        # Calculate score (-100 to 100)
        score = 0
        for sig in signals:
            if sig["strength"] == "strong_buy":
                score += 25
            elif sig["strength"] == "buy":
                score += 15
            elif sig["strength"] == "strong_sell":
                score -= 25
            elif sig["strength"] == "sell":
                score -= 15

        score = max(-100, min(100, score))

        return {
            "current_price": float(current_price),
            "indicators": {
                "rsi": float(latest_rsi) if not pd.isna(latest_rsi) else None,
                "macd": float(latest_macd) if not pd.isna(latest_macd) else None,
                "macd_signal": float(latest_signal) if not pd.isna(latest_signal) else None,
                "macd_histogram": float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else None,
                "sma_20": float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None,
                "sma_50": float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None,
                "sma_200": float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None,
                "bollinger_upper": float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
                "bollinger_lower": float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None,
                "atr": float(latest_atr) if not pd.isna(latest_atr) else None,
            },
            "support": float(support),
            "resistance": float(resistance),
            "trend": trend,
            "signals": signals,
            "score": score,
        }
