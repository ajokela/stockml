"""Main StockAnalyzer class - public API"""

from typing import Optional, Callable
import sys

from .data.yahoo import YahooFinanceClient
from .data.news import NewsClient, MockNewsClient
from .data.fmp import FMPClient, MockFMPClient
from .data.alphavantage import AlphaVantageClient, MockAlphaVantageClient
from .analysis.technical import TechnicalAnalyzer
from .analysis.fundamental import FundamentalAnalyzer
from .analysis.sentiment import SentimentAnalyzer
from .analysis.transcript import TranscriptAnalyzer
from .recommendation import RecommendationEngine, Action
from .report import ReportGenerator
from .cache import ReportCache


class StockAnalyzer:
    """Main class for analyzing stocks and generating recommendations

    Example usage:
        analyzer = StockAnalyzer(news_api_key="your_key", fmp_api_key="your_key")
        report = analyzer.analyze("AAPL")
        print(report["recommendation"]["action"])
    """

    def __init__(
        self,
        news_api_key: Optional[str] = None,
        fmp_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        alphavantage_api_key: Optional[str] = None,
        sentiment_weight: float = 0.25,
        technical_weight: float = 0.40,
        fundamental_weight: float = 0.35,
        cache_enabled: bool = True,
        cache_max_age_days: int = 7
    ):
        """Initialize StockAnalyzer

        Args:
            news_api_key: NewsAPI.org API key (optional, enables sentiment analysis)
            fmp_api_key: Financial Modeling Prep API key (optional, enhances fundamentals)
            openai_api_key: OpenAI API key (optional, enables AI-powered transcript summaries)
            alphavantage_api_key: Alpha Vantage API key (optional, provides news sentiment)
            sentiment_weight: Weight for sentiment in recommendations (0-1)
            technical_weight: Weight for technical analysis (0-1)
            fundamental_weight: Weight for fundamental analysis (0-1)
            cache_enabled: Whether to enable report caching (default True)
            cache_max_age_days: Maximum age of cached reports in days (default 7)
        """
        self.yahoo_client = YahooFinanceClient()

        # News client
        if news_api_key:
            self.news_client = NewsClient(api_key=news_api_key)
        else:
            self.news_client = MockNewsClient()

        # FMP client
        if fmp_api_key:
            self.fmp_client = FMPClient(api_key=fmp_api_key)
        else:
            self.fmp_client = MockFMPClient()

        # Alpha Vantage client
        if alphavantage_api_key:
            self.alphavantage_client = AlphaVantageClient(api_key=alphavantage_api_key)
        else:
            self.alphavantage_client = MockAlphaVantageClient()

        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.transcript_analyzer = TranscriptAnalyzer(openai_api_key=openai_api_key)

        self.openai_api_key = openai_api_key
        self._openai_client = None

        self.recommendation_engine = RecommendationEngine(
            sentiment_weight=sentiment_weight,
            technical_weight=technical_weight,
            fundamental_weight=fundamental_weight,
        )

        self.report_generator = ReportGenerator()

        # Cache
        self.cache_enabled = cache_enabled
        self.cache = ReportCache(max_age_days=cache_max_age_days) if cache_enabled else None

    def _fetch_fmp_data(self, ticker: str) -> dict:
        """Fetch enhanced data from Financial Modeling Prep

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with FMP data (dcf, rating, price_target, grades, insider_trades)
        """
        if not self.fmp_client.is_configured():
            return {}

        fmp_data = {}

        try:
            # DCF valuation
            fmp_data["dcf"] = self.fmp_client.get_dcf(ticker)
        except Exception:
            pass

        try:
            # FMP rating
            fmp_data["rating"] = self.fmp_client.get_rating(ticker)
        except Exception:
            pass

        try:
            # Price target summary
            fmp_data["price_target"] = self.fmp_client.get_price_target_summary(ticker)
        except Exception:
            pass

        try:
            # Analyst grades
            fmp_data["grades"] = self.fmp_client.get_grades(ticker, limit=10)
        except Exception:
            pass

        try:
            # Insider trading
            fmp_data["insider_trades"] = self.fmp_client.get_insider_trading(ticker, limit=20)
        except Exception:
            pass

        try:
            # Stock news from FMP (alternative to NewsAPI)
            fmp_data["news"] = self.fmp_client.get_stock_news(ticker, limit=10)
        except Exception:
            pass

        try:
            # Stock peers for comparison
            peer_symbols = self.fmp_client.get_stock_peers(ticker)
            if peer_symbols:
                peers = []
                for peer_ticker in peer_symbols[:5]:  # Limit to 5 peers
                    try:
                        quote = self.fmp_client.get_quote(peer_ticker)
                        if quote:
                            peers.append({
                                "ticker": peer_ticker,
                                "name": quote.get("name", peer_ticker),
                                "pe": quote.get("pe"),
                                "pb": quote.get("priceToBook"),
                                "market_cap": quote.get("marketCap"),
                                "div_yield": quote.get("dividendYield"),
                            })
                    except Exception:
                        continue
                fmp_data["peers"] = peers
        except Exception:
            pass

        return fmp_data

    def _fetch_alphavantage_data(self, ticker: str) -> dict:
        """Fetch data from Alpha Vantage

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with Alpha Vantage data (news_sentiment, company_overview, earnings)
        """
        if not self.alphavantage_client.is_configured():
            return {}

        av_data = {}

        try:
            # News sentiment with pre-computed scores
            sentiment_data = self.alphavantage_client.get_news_sentiment(ticker, limit=20)
            if sentiment_data and "feed" in sentiment_data:
                articles = []
                for item in sentiment_data["feed"][:10]:
                    # Find sentiment for this specific ticker
                    ticker_sentiment = None
                    for ts in item.get("ticker_sentiment", []):
                        if ts.get("ticker") == ticker:
                            ticker_sentiment = ts
                            break

                    articles.append({
                        "title": item.get("title"),
                        "summary": item.get("summary"),
                        "source": item.get("source"),
                        "url": item.get("url"),
                        "published_at": item.get("time_published"),
                        "overall_sentiment_score": float(item.get("overall_sentiment_score", 0)),
                        "overall_sentiment_label": item.get("overall_sentiment_label"),
                        "ticker_relevance": float(ticker_sentiment.get("relevance_score", 0)) if ticker_sentiment else 0,
                        "ticker_sentiment_score": float(ticker_sentiment.get("ticker_sentiment_score", 0)) if ticker_sentiment else 0,
                    })
                av_data["news_sentiment"] = articles

                # Calculate aggregate sentiment from Alpha Vantage scores
                if articles:
                    avg_sentiment = sum(a["ticker_sentiment_score"] for a in articles) / len(articles)
                    av_data["aggregate_sentiment"] = {
                        "score": avg_sentiment,
                        "label": "Bullish" if avg_sentiment > 0.1 else ("Bearish" if avg_sentiment < -0.1 else "Neutral"),
                        "articles_analyzed": len(articles)
                    }
        except Exception:
            pass

        try:
            # Company overview for additional fundamentals
            overview = self.alphavantage_client.get_company_overview(ticker)
            if overview:
                av_data["company_overview"] = {
                    "description": overview.get("Description"),
                    "sector": overview.get("Sector"),
                    "industry": overview.get("Industry"),
                    "market_cap": overview.get("MarketCapitalization"),
                    "pe_ratio": overview.get("PERatio"),
                    "peg_ratio": overview.get("PEGRatio"),
                    "book_value": overview.get("BookValue"),
                    "dividend_yield": overview.get("DividendYield"),
                    "eps": overview.get("EPS"),
                    "revenue_ttm": overview.get("RevenueTTM"),
                    "profit_margin": overview.get("ProfitMargin"),
                    "52_week_high": overview.get("52WeekHigh"),
                    "52_week_low": overview.get("52WeekLow"),
                    "analyst_target_price": overview.get("AnalystTargetPrice"),
                    "analyst_rating": overview.get("AnalystRatingStrongBuy"),
                }
        except Exception:
            pass

        try:
            # Earnings data
            earnings = self.alphavantage_client.get_earnings(ticker)
            if earnings:
                quarterly = earnings.get("quarterlyEarnings", [])[:4]
                av_data["earnings"] = {
                    "quarterly": [
                        {
                            "date": e.get("fiscalDateEnding"),
                            "reported_eps": e.get("reportedEPS"),
                            "estimated_eps": e.get("estimatedEPS"),
                            "surprise": e.get("surprise"),
                            "surprise_pct": e.get("surprisePercentage"),
                        }
                        for e in quarterly
                    ]
                }
        except Exception:
            pass

        return av_data

    def _fetch_transcript_analysis(self, ticker: str, company_name: str) -> Optional[dict]:
        """Fetch and analyze earnings transcript

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for context

        Returns:
            Transcript analysis dict or None if not available
        """
        if not self.fmp_client.is_configured():
            return None

        try:
            transcript_data = self.fmp_client.get_latest_transcript(ticker)
            if transcript_data and transcript_data.get("content"):
                analysis = self.transcript_analyzer.analyze(
                    transcript_data["content"],
                    company_name
                )
                # Add transcript metadata
                analysis["date"] = transcript_data.get("date")
                analysis["quarter"] = transcript_data.get("quarter")
                analysis["year"] = transcript_data.get("year")
                return analysis
        except Exception:
            pass

        return None

    def _get_openai_client(self):
        """Lazily initialize OpenAI client"""
        if self._openai_client is None and self.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except ImportError as e:
                import sys
                print(f"OpenAI import error: {e}", file=sys.stderr)
        return self._openai_client

    def _generate_narrative(
        self,
        ticker: str,
        company_info: dict,
        technical_analysis: Optional[dict],
        fundamental_analysis: Optional[dict],
        sentiment_analysis: Optional[dict],
        transcript_analysis: Optional[dict],
        recommendation: dict,
        current_price: float
    ) -> Optional[str]:
        """Generate an AI-powered investment narrative with 2-5 year horizon

        Args:
            ticker: Stock ticker symbol
            company_info: Company information dict
            technical_analysis: Technical analysis results
            fundamental_analysis: Fundamental analysis results
            sentiment_analysis: Sentiment analysis results
            transcript_analysis: Transcript analysis results
            recommendation: Recommendation dict
            current_price: Current stock price

        Returns:
            Narrative string or None if OpenAI not available
        """
        client = self._get_openai_client()
        if not client:
            return None

        # Build context from all analyses
        context_parts = []

        # Company info
        company_name = company_info.get("name", ticker)
        sector = company_info.get("sector", "Unknown")
        industry = company_info.get("industry", "Unknown")
        summary = company_info.get("summary", "")
        context_parts.append(f"Company: {company_name} ({ticker})")
        context_parts.append(f"Sector: {sector}, Industry: {industry}")
        if summary:
            context_parts.append(f"Business: {summary[:500]}")

        context_parts.append(f"\nCurrent Price: ${current_price:.2f}")

        # Recommendation
        context_parts.append(f"\nOverall Recommendation: {recommendation.get('action', 'HOLD')}")
        context_parts.append(f"Confidence: {recommendation.get('confidence', 50)}%")

        # Technical analysis
        if technical_analysis:
            context_parts.append(f"\n--- Technical Analysis ---")
            context_parts.append(f"Trend: {technical_analysis.get('trend', 'Unknown')}")
            context_parts.append(f"Technical Score: {technical_analysis.get('score', 0)}/100")
            indicators = technical_analysis.get("indicators", {})
            if indicators.get("rsi"):
                context_parts.append(f"RSI: {indicators['rsi']:.1f}")
            if technical_analysis.get("support"):
                context_parts.append(f"Support: ${technical_analysis['support']:.2f}")
            if technical_analysis.get("resistance"):
                context_parts.append(f"Resistance: ${technical_analysis['resistance']:.2f}")

        # Fundamental analysis
        if fundamental_analysis:
            metrics = fundamental_analysis.get("metrics", {})
            context_parts.append(f"\n--- Fundamental Analysis ---")
            context_parts.append(f"Fundamental Score: {fundamental_analysis.get('score', 0)}/100")
            if metrics.get("pe_ratio"):
                context_parts.append(f"P/E Ratio: {metrics['pe_ratio']:.1f}")
            if metrics.get("peg_ratio"):
                context_parts.append(f"PEG Ratio: {metrics['peg_ratio']:.2f}")
            if metrics.get("price_to_book"):
                context_parts.append(f"P/B Ratio: {metrics['price_to_book']:.2f}")
            if metrics.get("profit_margin"):
                context_parts.append(f"Profit Margin: {metrics['profit_margin']*100:.1f}%")
            if metrics.get("revenue_growth"):
                context_parts.append(f"Revenue Growth: {metrics['revenue_growth']*100:.1f}%")
            if metrics.get("debt_to_equity"):
                context_parts.append(f"Debt/Equity: {metrics['debt_to_equity']:.2f}")
            if metrics.get("dividend_yield"):
                context_parts.append(f"Dividend Yield: {metrics['dividend_yield']*100:.2f}%")

        # Sentiment analysis
        if sentiment_analysis:
            context_parts.append(f"\n--- News Sentiment ---")
            context_parts.append(f"Sentiment: {sentiment_analysis.get('classification', 'Unknown')}")
            context_parts.append(f"Sentiment Score: {sentiment_analysis.get('score', 0)}/100")
            context_parts.append(f"Articles Analyzed: {sentiment_analysis.get('articles_analyzed', 0)}")

        # Transcript analysis
        if transcript_analysis:
            context_parts.append(f"\n--- Earnings Call Analysis ---")
            context_parts.append(f"Outlook: {transcript_analysis.get('outlook', 'Unknown')}")
            context_parts.append(f"Transcript Score: {transcript_analysis.get('score', 0)}/100")
            if transcript_analysis.get("key_points"):
                context_parts.append("Key Points:")
                for point in transcript_analysis["key_points"][:3]:
                    context_parts.append(f"  - {point[:100]}")

        context = "\n".join(context_parts)

        prompt = f"""Based on the following stock analysis data, write a concise investment narrative
for {company_name} ({ticker}) with a 2-5 year investment horizon.

{context}

Write a 2-3 paragraph narrative that:
1. Summarizes the company's position and key investment thesis
2. Highlights the most important factors (positive and negative) for long-term investors
3. Provides perspective on the risk/reward profile over a 2-5 year timeframe

Be balanced, mention both opportunities and risks. Be specific about metrics when relevant.
Keep the tone professional and analytical. Do not use bullet points - write in flowing paragraphs."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional equity research analyst providing investment narratives for long-term investors."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Narrative generation error: {e}", file=sys.stderr)
            return None

    def analyze(
        self,
        ticker: str,
        period: str = "1y",
        news_days: int = 7,
        include_news: bool = True,
        include_fmp: bool = True,
        include_transcripts: bool = True,
        progress_callback: Callable[[int, Optional[str]], None] = None,
        force_refresh: bool = False
    ) -> dict:
        """Perform full analysis on a stock

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            period: Historical data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            news_days: Number of days of news to analyze
            include_news: Whether to fetch and analyze news
            include_fmp: Whether to fetch FMP enhanced data
            include_transcripts: Whether to fetch and analyze earnings transcripts
            progress_callback: Optional callback function(stage: int, message: str) for progress updates
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            JSON-serializable report dict with recommendation
        """
        def report_progress(stage: int, message: str = None):
            if progress_callback:
                progress_callback(stage, message)

        ticker = ticker.upper()

        # Check cache first (unless force_refresh is True)
        if self.cache and not force_refresh:
            cached_report = self.cache.get(ticker, period)
            if cached_report:
                # Report cache hit via progress callback
                report_progress(0, "Loading from cache...")
                report_progress(8, "Loaded from cache")
                return cached_report

        # Stage 0: Fetch data from Yahoo Finance
        report_progress(0)
        info = self.yahoo_client.get_stock_info(ticker)
        history = self.yahoo_client.get_price_history(ticker, period=period)
        dividends = self.yahoo_client.get_dividends(ticker)
        current_price = self.yahoo_client.get_current_price(ticker)
        company_name = info.get("shortName") or info.get("longName") or ticker

        # Stage 1: Fetch FMP enhanced data
        report_progress(1)
        fmp_data = {}
        if include_fmp:
            fmp_data = self._fetch_fmp_data(ticker)

        # Stage 2: Fetch Alpha Vantage data
        report_progress(2)
        av_data = self._fetch_alphavantage_data(ticker)

        # Stage 3: Perform technical analysis
        report_progress(3)
        technical_analysis = None
        if len(history) > 0:
            technical_analysis = self.technical_analyzer.analyze(history)

        # Stage 4: Perform fundamental analysis (now with FMP data)
        report_progress(4)
        fundamental_analysis = self.fundamental_analyzer.analyze(
            info,
            dividends,
            fmp_data=fmp_data if fmp_data else None
        )

        # Stage 5: Fetch news and sentiment
        report_progress(5)
        sentiment_analysis = None
        news_articles = []

        if include_news:
            # Try NewsAPI first, fall back to Alpha Vantage, then FMP news
            if self.news_client.is_configured():
                news_articles = self.news_client.fetch_stock_news(
                    ticker,
                    company_name=company_name,
                    days=news_days
                )
            elif av_data.get("news_sentiment"):
                # Use Alpha Vantage news (already has sentiment scores)
                news_articles = [
                    {
                        "title": article.get("title"),
                        "description": article.get("summary"),
                        "source": article.get("source"),
                        "url": article.get("url"),
                        "published_at": article.get("published_at"),
                    }
                    for article in av_data["news_sentiment"]
                ]
            elif fmp_data.get("news"):
                # Use FMP news as fallback
                news_articles = [
                    {
                        "title": article.get("title"),
                        "description": article.get("text"),
                        "source": article.get("site"),
                        "url": article.get("url"),
                        "published_at": article.get("publishedDate"),
                    }
                    for article in fmp_data["news"]
                ]

            if news_articles:
                sentiment_analysis = self.sentiment_analyzer.analyze(news_articles)

            # Enhance sentiment with Alpha Vantage pre-computed scores if available
            if av_data.get("aggregate_sentiment") and sentiment_analysis:
                av_sentiment = av_data["aggregate_sentiment"]
                # Blend our sentiment with Alpha Vantage (AV is pre-computed ML-based)
                av_score_normalized = av_sentiment["score"] * 100  # Convert -1 to 1 range to -100 to 100
                blended_score = (sentiment_analysis.get("score", 0) + av_score_normalized) / 2
                sentiment_analysis["score"] = int(blended_score)
                sentiment_analysis["alphavantage_sentiment"] = av_sentiment

        # Stage 6: Analyze earnings transcripts
        report_progress(6)
        transcript_analysis = None
        if include_transcripts:
            transcript_analysis = self._fetch_transcript_analysis(ticker, company_name)

        # Generate recommendation
        recommendation = self.recommendation_engine.generate_recommendation(
            technical_analysis=technical_analysis,
            fundamental_analysis=fundamental_analysis,
            sentiment_analysis=sentiment_analysis,
            transcript_analysis=transcript_analysis,
            current_price=current_price,
            fmp_data=fmp_data,
        )

        # Generate report
        report = self.report_generator.generate_report(
            ticker=ticker,
            current_price=current_price,
            recommendation=self.recommendation_engine.to_dict(recommendation),
            technical_analysis=technical_analysis,
            fundamental_analysis=fundamental_analysis,
            sentiment_analysis=sentiment_analysis,
            transcript_analysis=transcript_analysis,
            news_articles=news_articles,
        )

        # Add FMP-specific data to report
        if fmp_data:
            report["fmp_data"] = {
                "dcf_value": fmp_data.get("dcf", {}).get("dcf"),
                "fmp_rating": fmp_data.get("rating", {}).get("rating"),
                "fmp_recommendation": fmp_data.get("rating", {}).get("ratingRecommendation"),
                "analyst_target": fmp_data.get("price_target", {}).get("targetConsensus"),
                "peers": fmp_data.get("peers", []),
            }

        # Add Alpha Vantage data to report
        if av_data:
            report["alphavantage_data"] = {
                "earnings": av_data.get("earnings"),
                "sentiment": av_data.get("aggregate_sentiment"),
            }

        # Stage 7: Generate AI narrative
        report_progress(7)
        company_info = fundamental_analysis.get("company", {}) if fundamental_analysis else {}
        narrative = self._generate_narrative(
            ticker=ticker,
            company_info=company_info,
            technical_analysis=technical_analysis,
            fundamental_analysis=fundamental_analysis,
            sentiment_analysis=sentiment_analysis,
            transcript_analysis=transcript_analysis,
            recommendation=self.recommendation_engine.to_dict(recommendation),
            current_price=current_price
        )
        if narrative:
            report["investment_narrative"] = narrative

        # Stage 8: Finalize report
        report_progress(8)

        # Store in cache
        if self.cache:
            self.cache.set(ticker, report, period)

        return report

    def quick_recommendation(self, ticker: str) -> str:
        """Get a simple recommendation string

        Args:
            ticker: Stock ticker symbol

        Returns:
            Recommendation string (e.g., "BUY", "SELL", "HOLD")
        """
        report = self.analyze(ticker)
        return report["recommendation"]["action"]

    def get_technical_analysis(self, ticker: str, period: str = "1y") -> dict:
        """Get only technical analysis for a stock

        Args:
            ticker: Stock ticker symbol
            period: Historical data period

        Returns:
            Technical analysis results dict
        """
        history = self.yahoo_client.get_price_history(ticker, period=period)
        return self.technical_analyzer.analyze(history)

    def get_fundamental_analysis(self, ticker: str, include_fmp: bool = True) -> dict:
        """Get only fundamental analysis for a stock

        Args:
            ticker: Stock ticker symbol
            include_fmp: Whether to include FMP enhanced data

        Returns:
            Fundamental analysis results dict
        """
        info = self.yahoo_client.get_stock_info(ticker)
        dividends = self.yahoo_client.get_dividends(ticker)

        fmp_data = None
        if include_fmp:
            fmp_data = self._fetch_fmp_data(ticker)

        return self.fundamental_analyzer.analyze(info, dividends, fmp_data=fmp_data)

    def get_sentiment_analysis(self, ticker: str, days: int = 7) -> dict:
        """Get only sentiment analysis for a stock

        Args:
            ticker: Stock ticker symbol
            days: Number of days of news to analyze

        Returns:
            Sentiment analysis results dict

        Raises:
            ValueError: If no news source is configured
        """
        news_articles = []

        if self.news_client.is_configured():
            info = self.yahoo_client.get_stock_info(ticker)
            company_name = info.get("shortName") or info.get("longName")
            news_articles = self.news_client.fetch_stock_news(
                ticker,
                company_name=company_name,
                days=days
            )
        elif self.fmp_client.is_configured():
            fmp_news = self.fmp_client.get_stock_news(ticker, limit=20)
            news_articles = [
                {
                    "title": article.get("title"),
                    "description": article.get("text"),
                    "source": article.get("site"),
                    "url": article.get("url"),
                    "published_at": article.get("publishedDate"),
                }
                for article in fmp_news
            ]
        else:
            raise ValueError("News API or FMP API key required for sentiment analysis")

        return self.sentiment_analyzer.analyze(news_articles)

    def get_transcript_analysis(self, ticker: str) -> dict:
        """Get earnings transcript analysis for a stock

        Args:
            ticker: Stock ticker symbol

        Returns:
            Transcript analysis dict

        Raises:
            ValueError: If FMP API key is not configured or no transcript available
        """
        if not self.fmp_client.is_configured():
            raise ValueError("FMP API key required for transcript analysis")

        info = self.yahoo_client.get_stock_info(ticker)
        company_name = info.get("shortName") or info.get("longName") or ticker

        transcript_data = self.fmp_client.get_latest_transcript(ticker)
        if not transcript_data or not transcript_data.get("content"):
            raise ValueError(f"No earnings transcript available for {ticker}")

        analysis = self.transcript_analyzer.analyze(
            transcript_data["content"],
            company_name
        )
        analysis["date"] = transcript_data.get("date")
        analysis["quarter"] = transcript_data.get("quarter")
        analysis["year"] = transcript_data.get("year")

        return analysis

    def get_earnings_transcripts(self, ticker: str, limit: int = 4) -> list:
        """Get raw earnings transcripts from FMP

        Args:
            ticker: Stock ticker symbol
            limit: Number of transcripts to fetch

        Returns:
            List of transcript dicts
        """
        if not self.fmp_client.is_configured():
            raise ValueError("FMP API key required for transcripts")

        transcripts = self.fmp_client.get_earnings_transcripts(ticker)
        return transcripts[:limit] if transcripts else []

    def get_fmp_data(self, ticker: str) -> dict:
        """Get raw FMP data for a stock

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with all FMP data

        Raises:
            ValueError: If FMP API key is not configured
        """
        if not self.fmp_client.is_configured():
            raise ValueError("FMP API key required")

        return self._fetch_fmp_data(ticker)

    def get_dcf_valuation(self, ticker: str) -> dict:
        """Get DCF valuation from FMP

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with DCF value and stock price
        """
        if not self.fmp_client.is_configured():
            raise ValueError("FMP API key required for DCF valuation")

        return self.fmp_client.get_dcf(ticker)

    def get_analyst_ratings(self, ticker: str) -> dict:
        """Get analyst ratings and price targets from FMP

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with price targets, grades, and recommendations
        """
        if not self.fmp_client.is_configured():
            raise ValueError("FMP API key required for analyst ratings")

        return {
            "price_target": self.fmp_client.get_price_target_summary(ticker),
            "grades": self.fmp_client.get_grades(ticker, limit=10),
            "rating": self.fmp_client.get_rating(ticker),
        }

    def get_insider_trading(self, ticker: str, limit: int = 20) -> list:
        """Get insider trading activity from FMP

        Args:
            ticker: Stock ticker symbol
            limit: Number of trades to fetch

        Returns:
            List of insider trades
        """
        if not self.fmp_client.is_configured():
            raise ValueError("FMP API key required for insider trading data")

        return self.fmp_client.get_insider_trading(ticker, limit=limit)

    def compare_stocks(self, tickers: list, period: str = "1y") -> list:
        """Compare multiple stocks and rank them

        Args:
            tickers: List of ticker symbols
            period: Historical data period

        Returns:
            List of reports sorted by recommendation score
        """
        reports = []
        for ticker in tickers:
            report = self.analyze(ticker, period=period)
            reports.append(report)

        # Sort by confidence (descending) for BUY recommendations
        def sort_key(report):
            action = report["recommendation"]["action"]
            confidence = report["recommendation"]["confidence"]

            # Prioritize: STRONG_BUY > BUY > HOLD > SELL > STRONG_SELL
            action_scores = {
                "STRONG_BUY": 5,
                "BUY": 4,
                "HOLD": 3,
                "SELL": 2,
                "STRONG_SELL": 1,
            }
            return (action_scores.get(action, 0), confidence)

        reports.sort(key=sort_key, reverse=True)
        return reports
