"""Sentiment analysis for news articles"""

from datetime import datetime
from typing import List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """Analyze sentiment of news articles using VADER"""

    def __init__(self):
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> dict:
        """Analyze sentiment of a single piece of text

        Args:
            text: Text to analyze

        Returns:
            Dict with sentiment scores:
            - compound: Overall sentiment (-1 to 1)
            - positive: Positive sentiment (0 to 1)
            - negative: Negative sentiment (0 to 1)
            - neutral: Neutral sentiment (0 to 1)
        """
        scores = self._analyzer.polarity_scores(text)
        return {
            "compound": scores["compound"],
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }

    def analyze_article(self, article: dict) -> dict:
        """Analyze sentiment of a news article

        Args:
            article: Article dict with title, description, content

        Returns:
            Dict with sentiment analysis results
        """
        # Combine title and description for analysis
        # (content is often truncated by NewsAPI)
        text_parts = []

        title = article.get("title")
        if title:
            text_parts.append(title)

        description = article.get("description")
        if description:
            text_parts.append(description)

        if not text_parts:
            return {
                "sentiment": {"compound": 0, "positive": 0, "negative": 0, "neutral": 1},
                "classification": "neutral",
            }

        combined_text = " ".join(text_parts)
        sentiment = self.analyze_text(combined_text)

        # Classify based on compound score
        compound = sentiment["compound"]
        if compound >= 0.05:
            classification = "positive"
        elif compound <= -0.05:
            classification = "negative"
        else:
            classification = "neutral"

        return {
            "sentiment": sentiment,
            "classification": classification,
            "title": title,
            "source": article.get("source"),
            "published_at": article.get("published_at"),
        }

    def analyze_articles(
        self,
        articles: List[dict],
        recency_weight: bool = True
    ) -> dict:
        """Analyze sentiment across multiple articles

        Args:
            articles: List of article dicts
            recency_weight: If True, weight recent articles more heavily

        Returns:
            Dict with aggregated sentiment analysis
        """
        if not articles:
            return {
                "overall_sentiment": 0,
                "classification": "neutral",
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "articles_analyzed": 0,
                "article_sentiments": [],
            }

        article_results = []
        weights = []

        for i, article in enumerate(articles):
            result = self.analyze_article(article)
            article_results.append(result)

            # Calculate recency weight (more recent = higher weight)
            if recency_weight:
                # Assume articles are sorted newest first
                # Weight decays from 1.0 to 0.5 over the list
                weight = 1.0 - (i / len(articles)) * 0.5
            else:
                weight = 1.0

            weights.append(weight)

        # Calculate weighted average sentiment
        total_weight = sum(weights)
        weighted_sentiment = sum(
            result["sentiment"]["compound"] * weight
            for result, weight in zip(article_results, weights)
        ) / total_weight

        # Count classifications
        positive_count = sum(1 for r in article_results if r["classification"] == "positive")
        negative_count = sum(1 for r in article_results if r["classification"] == "negative")
        neutral_count = sum(1 for r in article_results if r["classification"] == "neutral")

        # Overall classification
        if weighted_sentiment >= 0.05:
            overall_class = "positive"
        elif weighted_sentiment <= -0.05:
            overall_class = "negative"
        else:
            overall_class = "neutral"

        # Calculate score (-100 to 100)
        score = int(weighted_sentiment * 100)

        return {
            "overall_sentiment": weighted_sentiment,
            "classification": overall_class,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "articles_analyzed": len(articles),
            "article_sentiments": article_results,
            "score": score,
        }

    def analyze(self, articles: List[dict]) -> dict:
        """Convenience method - alias for analyze_articles"""
        return self.analyze_articles(articles)
