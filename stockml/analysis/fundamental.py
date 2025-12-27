"""Fundamental analysis of stocks"""

from typing import Optional, List
import pandas as pd


class FundamentalAnalyzer:
    """Analyze fundamental metrics like P/E, dividends, and financials"""

    # Benchmark values for scoring
    BENCHMARKS = {
        "pe_ratio": {"low": 15, "high": 25},
        "peg_ratio": {"low": 1.0, "high": 2.0},
        "price_to_book": {"low": 1.0, "high": 3.0},
        "debt_to_equity": {"low": 0.5, "high": 1.5},
        "current_ratio": {"low": 1.0, "high": 2.0},
        "profit_margin": {"low": 0.10, "high": 0.20},
        "dividend_yield": {"low": 0.02, "high": 0.05},
    }

    def analyze_valuation(self, info: dict) -> dict:
        """Analyze valuation metrics

        Args:
            info: Stock info dict from Yahoo Finance

        Returns:
            Dict with valuation metrics and scores
        """
        metrics = {}
        signals = []

        # P/E Ratio
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe:
            metrics["pe_ratio"] = pe
            if pe < self.BENCHMARKS["pe_ratio"]["low"]:
                signals.append({"metric": "pe_ratio", "signal": "undervalued", "strength": "buy"})
            elif pe > self.BENCHMARKS["pe_ratio"]["high"]:
                signals.append({"metric": "pe_ratio", "signal": "overvalued", "strength": "sell"})

        # PEG Ratio
        peg = info.get("pegRatio")
        if peg:
            metrics["peg_ratio"] = peg
            if peg < self.BENCHMARKS["peg_ratio"]["low"]:
                signals.append({"metric": "peg_ratio", "signal": "undervalued", "strength": "buy"})
            elif peg > self.BENCHMARKS["peg_ratio"]["high"]:
                signals.append({"metric": "peg_ratio", "signal": "overvalued", "strength": "sell"})

        # Price to Book
        pb = info.get("priceToBook")
        if pb:
            metrics["price_to_book"] = pb
            if pb < self.BENCHMARKS["price_to_book"]["low"]:
                signals.append({"metric": "price_to_book", "signal": "undervalued", "strength": "buy"})
            elif pb > self.BENCHMARKS["price_to_book"]["high"]:
                signals.append({"metric": "price_to_book", "signal": "overvalued", "strength": "sell"})

        # Market Cap
        market_cap = info.get("marketCap")
        if market_cap:
            metrics["market_cap"] = market_cap
            if market_cap > 200_000_000_000:
                metrics["market_cap_category"] = "mega_cap"
            elif market_cap > 10_000_000_000:
                metrics["market_cap_category"] = "large_cap"
            elif market_cap > 2_000_000_000:
                metrics["market_cap_category"] = "mid_cap"
            elif market_cap > 300_000_000:
                metrics["market_cap_category"] = "small_cap"
            else:
                metrics["market_cap_category"] = "micro_cap"

        return {"metrics": metrics, "signals": signals}

    def analyze_financial_health(self, info: dict) -> dict:
        """Analyze financial health indicators"""
        metrics = {}
        signals = []

        # Debt to Equity
        de = info.get("debtToEquity")
        if de:
            de_ratio = de / 100 if de > 10 else de  # Normalize if percentage
            metrics["debt_to_equity"] = de_ratio
            if de_ratio < self.BENCHMARKS["debt_to_equity"]["low"]:
                signals.append({"metric": "debt_to_equity", "signal": "low_debt", "strength": "buy"})
            elif de_ratio > self.BENCHMARKS["debt_to_equity"]["high"]:
                signals.append({"metric": "debt_to_equity", "signal": "high_debt", "strength": "sell"})

        # Current Ratio
        cr = info.get("currentRatio")
        if cr:
            metrics["current_ratio"] = cr
            if cr > self.BENCHMARKS["current_ratio"]["high"]:
                signals.append({"metric": "current_ratio", "signal": "strong_liquidity", "strength": "buy"})
            elif cr < self.BENCHMARKS["current_ratio"]["low"]:
                signals.append({"metric": "current_ratio", "signal": "weak_liquidity", "strength": "sell"})

        # Quick Ratio
        qr = info.get("quickRatio")
        if qr:
            metrics["quick_ratio"] = qr

        return {"metrics": metrics, "signals": signals}

    def analyze_profitability(self, info: dict) -> dict:
        """Analyze profitability metrics"""
        metrics = {}
        signals = []

        # Profit Margin
        pm = info.get("profitMargins")
        if pm:
            metrics["profit_margin"] = pm
            if pm > self.BENCHMARKS["profit_margin"]["high"]:
                signals.append({"metric": "profit_margin", "signal": "high_profitability", "strength": "buy"})
            elif pm < self.BENCHMARKS["profit_margin"]["low"]:
                signals.append({"metric": "profit_margin", "signal": "low_profitability", "strength": "sell"})

        # Operating Margin
        om = info.get("operatingMargins")
        if om:
            metrics["operating_margin"] = om

        # Return on Equity
        roe = info.get("returnOnEquity")
        if roe:
            metrics["return_on_equity"] = roe
            if roe > 0.15:
                signals.append({"metric": "roe", "signal": "strong_returns", "strength": "buy"})
            elif roe < 0.05:
                signals.append({"metric": "roe", "signal": "weak_returns", "strength": "sell"})

        # Return on Assets
        roa = info.get("returnOnAssets")
        if roa:
            metrics["return_on_assets"] = roa

        # Revenue Growth
        rg = info.get("revenueGrowth")
        if rg:
            metrics["revenue_growth"] = rg
            if rg > 0.15:
                signals.append({"metric": "revenue_growth", "signal": "strong_growth", "strength": "buy"})
            elif rg < 0:
                signals.append({"metric": "revenue_growth", "signal": "declining_revenue", "strength": "sell"})

        # Earnings Growth
        eg = info.get("earningsGrowth")
        if eg:
            metrics["earnings_growth"] = eg

        return {"metrics": metrics, "signals": signals}

    def analyze_dividends(
        self,
        info: dict,
        dividends: Optional[pd.Series] = None,
        current_price: Optional[float] = None
    ) -> dict:
        """Analyze dividend metrics"""
        metrics = {}
        signals = []

        # Dividend Yield
        div_yield = info.get("dividendYield")
        if div_yield:
            metrics["dividend_yield"] = div_yield
            if div_yield > self.BENCHMARKS["dividend_yield"]["high"]:
                signals.append({"metric": "dividend_yield", "signal": "high_yield", "strength": "buy"})
            elif div_yield > self.BENCHMARKS["dividend_yield"]["low"]:
                signals.append({"metric": "dividend_yield", "signal": "moderate_yield", "strength": "neutral"})

        # Dividend Rate
        div_rate = info.get("dividendRate")
        if div_rate:
            metrics["dividend_rate"] = div_rate

        # Payout Ratio
        payout = info.get("payoutRatio")
        if payout:
            metrics["payout_ratio"] = payout
            if payout > 0.8:
                signals.append({"metric": "payout_ratio", "signal": "high_payout", "strength": "sell"})
            elif payout < 0.4 and payout > 0:
                signals.append({"metric": "payout_ratio", "signal": "sustainable_payout", "strength": "buy"})

        # Analyze dividend history if provided
        if dividends is not None and len(dividends) > 0:
            metrics["has_dividends"] = True
            metrics["dividend_count"] = len(dividends)

            # Calculate dividend growth
            if len(dividends) >= 4:
                recent = dividends.tail(4).sum()
                older = dividends.iloc[-8:-4].sum() if len(dividends) >= 8 else None
                if older and older > 0:
                    growth = (recent - older) / older
                    metrics["dividend_growth"] = growth
                    if growth > 0.05:
                        signals.append({"metric": "dividend_growth", "signal": "growing_dividends", "strength": "buy"})
        else:
            metrics["has_dividends"] = False

        return {"metrics": metrics, "signals": signals}

    def analyze_dcf(self, dcf_data: dict, current_price: float) -> dict:
        """Analyze DCF (Discounted Cash Flow) valuation from FMP

        Args:
            dcf_data: DCF data from FMP API
            current_price: Current stock price

        Returns:
            Dict with DCF metrics and signals
        """
        metrics = {}
        signals = []

        dcf_value = dcf_data.get("dcf")
        if dcf_value and current_price:
            metrics["dcf_value"] = dcf_value
            metrics["dcf_upside"] = ((dcf_value - current_price) / current_price) * 100

            # Signal based on DCF vs current price
            if dcf_value > current_price * 1.20:  # >20% undervalued
                signals.append({
                    "metric": "dcf",
                    "signal": "significantly_undervalued",
                    "strength": "strong_buy"
                })
            elif dcf_value > current_price * 1.10:  # >10% undervalued
                signals.append({
                    "metric": "dcf",
                    "signal": "undervalued",
                    "strength": "buy"
                })
            elif dcf_value < current_price * 0.80:  # >20% overvalued
                signals.append({
                    "metric": "dcf",
                    "signal": "significantly_overvalued",
                    "strength": "strong_sell"
                })
            elif dcf_value < current_price * 0.90:  # >10% overvalued
                signals.append({
                    "metric": "dcf",
                    "signal": "overvalued",
                    "strength": "sell"
                })

        return {"metrics": metrics, "signals": signals}

    def analyze_analyst_ratings(
        self,
        price_target: dict,
        grades: List[dict],
        current_price: float
    ) -> dict:
        """Analyze analyst price targets and ratings from FMP

        Args:
            price_target: Price target summary from FMP
            grades: Recent analyst grade changes
            current_price: Current stock price

        Returns:
            Dict with analyst metrics and signals
        """
        metrics = {}
        signals = []

        # Price target analysis
        if price_target:
            target_high = price_target.get("targetHigh")
            target_low = price_target.get("targetLow")
            target_avg = price_target.get("targetConsensus") or price_target.get("targetMedian")
            num_analysts = price_target.get("numberOfAnalysts")

            if target_avg:
                metrics["analyst_target_avg"] = target_avg
                metrics["analyst_target_high"] = target_high
                metrics["analyst_target_low"] = target_low
                metrics["analyst_count"] = num_analysts

                if current_price:
                    upside = ((target_avg - current_price) / current_price) * 100
                    metrics["analyst_upside"] = upside

                    if upside > 25:
                        signals.append({
                            "metric": "analyst_target",
                            "signal": "strong_upside",
                            "strength": "strong_buy"
                        })
                    elif upside > 10:
                        signals.append({
                            "metric": "analyst_target",
                            "signal": "upside_potential",
                            "strength": "buy"
                        })
                    elif upside < -20:
                        signals.append({
                            "metric": "analyst_target",
                            "signal": "strong_downside",
                            "strength": "strong_sell"
                        })
                    elif upside < -10:
                        signals.append({
                            "metric": "analyst_target",
                            "signal": "downside_risk",
                            "strength": "sell"
                        })

        # Recent grade changes
        if grades:
            upgrades = 0
            downgrades = 0
            for grade in grades[:10]:  # Last 10 grades
                new_grade = grade.get("newGrade", "").lower()
                prev_grade = grade.get("previousGrade", "").lower()

                buy_grades = ["buy", "strong buy", "outperform", "overweight"]
                sell_grades = ["sell", "strong sell", "underperform", "underweight"]

                if any(g in new_grade for g in buy_grades):
                    if any(g in prev_grade for g in sell_grades) or "hold" in prev_grade:
                        upgrades += 1
                elif any(g in new_grade for g in sell_grades):
                    if any(g in prev_grade for g in buy_grades) or "hold" in prev_grade:
                        downgrades += 1

            metrics["recent_upgrades"] = upgrades
            metrics["recent_downgrades"] = downgrades

            if upgrades > downgrades + 2:
                signals.append({
                    "metric": "analyst_grades",
                    "signal": "upgrade_momentum",
                    "strength": "buy"
                })
            elif downgrades > upgrades + 2:
                signals.append({
                    "metric": "analyst_grades",
                    "signal": "downgrade_momentum",
                    "strength": "sell"
                })

        return {"metrics": metrics, "signals": signals}

    def analyze_insider_trading(self, insider_trades: List[dict]) -> dict:
        """Analyze insider trading activity from FMP

        Args:
            insider_trades: List of insider trades from FMP

        Returns:
            Dict with insider trading metrics and signals
        """
        metrics = {}
        signals = []

        if not insider_trades:
            return {"metrics": metrics, "signals": signals}

        buy_value = 0
        sell_value = 0
        buy_count = 0
        sell_count = 0

        for trade in insider_trades:
            transaction_type = trade.get("transactionType", "").lower()
            value = abs(trade.get("securitiesTransacted", 0) * trade.get("price", 0))

            if "purchase" in transaction_type or "buy" in transaction_type:
                buy_value += value
                buy_count += 1
            elif "sale" in transaction_type or "sell" in transaction_type:
                sell_value += value
                sell_count += 1

        metrics["insider_buy_count"] = buy_count
        metrics["insider_sell_count"] = sell_count
        metrics["insider_buy_value"] = buy_value
        metrics["insider_sell_value"] = sell_value

        # Net insider sentiment
        if buy_value > 0 or sell_value > 0:
            net_ratio = (buy_value - sell_value) / (buy_value + sell_value)
            metrics["insider_sentiment"] = net_ratio

            if buy_count > sell_count and buy_value > sell_value * 2:
                signals.append({
                    "metric": "insider_trading",
                    "signal": "insider_buying",
                    "strength": "buy"
                })
            elif sell_count > buy_count * 2 and sell_value > buy_value * 3:
                signals.append({
                    "metric": "insider_trading",
                    "signal": "insider_selling",
                    "strength": "sell"
                })

        return {"metrics": metrics, "signals": signals}

    def analyze_fmp_rating(self, rating_data: dict) -> dict:
        """Analyze FMP's proprietary stock rating

        Args:
            rating_data: Rating data from FMP API

        Returns:
            Dict with FMP rating metrics and signals
        """
        metrics = {}
        signals = []

        if not rating_data:
            return {"metrics": metrics, "signals": signals}

        rating = rating_data.get("rating")
        rating_score = rating_data.get("ratingScore")
        recommendation = rating_data.get("ratingRecommendation")

        if rating:
            metrics["fmp_rating"] = rating
            metrics["fmp_rating_score"] = rating_score
            metrics["fmp_recommendation"] = recommendation

            # Component scores
            metrics["fmp_dcf_score"] = rating_data.get("ratingDetailsDCFScore")
            metrics["fmp_roe_score"] = rating_data.get("ratingDetailsROEScore")
            metrics["fmp_roa_score"] = rating_data.get("ratingDetailsROAScore")
            metrics["fmp_pe_score"] = rating_data.get("ratingDetailsPEScore")
            metrics["fmp_pb_score"] = rating_data.get("ratingDetailsPBScore")

            if rating in ["S", "A"]:
                signals.append({
                    "metric": "fmp_rating",
                    "signal": f"strong_rating_{rating}",
                    "strength": "strong_buy"
                })
            elif rating == "B":
                signals.append({
                    "metric": "fmp_rating",
                    "signal": "good_rating_B",
                    "strength": "buy"
                })
            elif rating in ["D", "F"]:
                signals.append({
                    "metric": "fmp_rating",
                    "signal": f"weak_rating_{rating}",
                    "strength": "sell"
                })

        return {"metrics": metrics, "signals": signals}

    def analyze(
        self,
        info: dict,
        dividends: Optional[pd.Series] = None,
        fmp_data: Optional[dict] = None
    ) -> dict:
        """Perform full fundamental analysis

        Args:
            info: Stock info dict from Yahoo Finance
            dividends: Optional dividend history Series
            fmp_data: Optional FMP data dict with keys:
                - dcf: DCF valuation data
                - rating: FMP rating data
                - price_target: Price target summary
                - grades: Analyst grade changes
                - insider_trades: Insider trading activity

        Returns:
            Dict with all fundamental metrics, signals, and score
        """
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        # Yahoo Finance based analysis
        valuation = self.analyze_valuation(info)
        health = self.analyze_financial_health(info)
        profitability = self.analyze_profitability(info)
        dividend_analysis = self.analyze_dividends(info, dividends, current_price)

        # Combine all metrics and signals
        all_metrics = {
            **valuation["metrics"],
            **health["metrics"],
            **profitability["metrics"],
            **dividend_analysis["metrics"],
        }

        all_signals = (
            valuation["signals"]
            + health["signals"]
            + profitability["signals"]
            + dividend_analysis["signals"]
        )

        # FMP enhanced analysis
        if fmp_data:
            # DCF valuation
            if fmp_data.get("dcf") and current_price:
                dcf_analysis = self.analyze_dcf(fmp_data["dcf"], current_price)
                all_metrics.update(dcf_analysis["metrics"])
                all_signals.extend(dcf_analysis["signals"])

            # Analyst ratings
            if fmp_data.get("price_target") or fmp_data.get("grades"):
                analyst_analysis = self.analyze_analyst_ratings(
                    fmp_data.get("price_target", {}),
                    fmp_data.get("grades", []),
                    current_price
                )
                all_metrics.update(analyst_analysis["metrics"])
                all_signals.extend(analyst_analysis["signals"])

            # Insider trading
            if fmp_data.get("insider_trades"):
                insider_analysis = self.analyze_insider_trading(fmp_data["insider_trades"])
                all_metrics.update(insider_analysis["metrics"])
                all_signals.extend(insider_analysis["signals"])

            # FMP rating
            if fmp_data.get("rating"):
                rating_analysis = self.analyze_fmp_rating(fmp_data["rating"])
                all_metrics.update(rating_analysis["metrics"])
                all_signals.extend(rating_analysis["signals"])

        # Calculate score (-100 to 100)
        score = 0
        for sig in all_signals:
            strength = sig["strength"]
            if strength == "strong_buy":
                score += 20
            elif strength == "buy":
                score += 12
            elif strength == "strong_sell":
                score -= 20
            elif strength == "sell":
                score -= 12
            # neutral signals don't affect score

        score = max(-100, min(100, score))

        # Add company info
        company_info = {
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            "summary": info.get("longBusinessSummary"),
        }

        return {
            "company": company_info,
            "metrics": all_metrics,
            "signals": all_signals,
            "score": score,
        }
