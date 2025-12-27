"""Report generation for stock analysis"""

import json
from datetime import datetime
from typing import Optional


class ReportGenerator:
    """Generate JSON reports from analysis results"""

    def generate_report(
        self,
        ticker: str,
        current_price: Optional[float],
        recommendation: dict,
        technical_analysis: Optional[dict] = None,
        fundamental_analysis: Optional[dict] = None,
        sentiment_analysis: Optional[dict] = None,
        transcript_analysis: Optional[dict] = None,
        news_articles: Optional[list] = None
    ) -> dict:
        """Generate a comprehensive analysis report

        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            recommendation: Recommendation dict from RecommendationEngine
            technical_analysis: Results from TechnicalAnalyzer
            fundamental_analysis: Results from FundamentalAnalyzer
            sentiment_analysis: Results from SentimentAnalyzer
            transcript_analysis: Results from TranscriptAnalyzer
            news_articles: List of news articles analyzed

        Returns:
            JSON-serializable report dict
        """
        report = {
            "ticker": ticker.upper(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "current_price": current_price,
            "recommendation": recommendation,
        }

        # Analysis section
        analysis = {}

        if technical_analysis:
            analysis["technical"] = {
                "trend": technical_analysis.get("trend"),
                "score": technical_analysis.get("score"),
                "indicators": technical_analysis.get("indicators"),
                "support": technical_analysis.get("support"),
                "resistance": technical_analysis.get("resistance"),
                "signals": technical_analysis.get("signals"),
            }

        if fundamental_analysis:
            analysis["fundamental"] = {
                "company": fundamental_analysis.get("company"),
                "score": fundamental_analysis.get("score"),
                "metrics": fundamental_analysis.get("metrics"),
                "signals": fundamental_analysis.get("signals"),
            }

        if sentiment_analysis:
            analysis["sentiment"] = {
                "score": sentiment_analysis.get("score"),
                "overall_sentiment": sentiment_analysis.get("overall_sentiment"),
                "classification": sentiment_analysis.get("classification"),
                "articles_analyzed": sentiment_analysis.get("articles_analyzed"),
                "positive_count": sentiment_analysis.get("positive_count"),
                "negative_count": sentiment_analysis.get("negative_count"),
                "neutral_count": sentiment_analysis.get("neutral_count"),
            }

        if transcript_analysis:
            sentiment_data = transcript_analysis.get("sentiment", {})
            analysis["transcript"] = {
                "date": transcript_analysis.get("date"),
                "quarter": transcript_analysis.get("quarter"),
                "year": transcript_analysis.get("year"),
                "score": transcript_analysis.get("score"),
                "outlook": transcript_analysis.get("outlook"),
                "sentiment_score": sentiment_data.get("sentiment_score"),
                "confidence": sentiment_data.get("confidence"),
                "bullish_keywords": sentiment_data.get("bullish_count"),
                "bearish_keywords": sentiment_data.get("bearish_count"),
                "summary": transcript_analysis.get("summary"),
                "key_points": transcript_analysis.get("key_points"),
                "metrics": transcript_analysis.get("metrics"),
                "summary_method": transcript_analysis.get("summary_method"),
            }

        report["analysis"] = analysis

        # News summary (top articles)
        if news_articles:
            report["news_summary"] = [
                {
                    "title": article.get("title"),
                    "source": article.get("source"),
                    "published_at": article.get("published_at"),
                    "url": article.get("url"),
                    "description": article.get("description"),
                }
                for article in news_articles[:5]  # Top 5 articles
            ]

        return report

    def to_json(self, report: dict, indent: int = 2) -> str:
        """Convert report to JSON string"""
        return json.dumps(report, indent=indent, default=str)

    def save_report(self, report: dict, filepath: str) -> None:
        """Save report to a JSON file"""
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
