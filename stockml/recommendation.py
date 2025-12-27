"""Recommendation engine combining all analyses"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Action(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class Recommendation:
    """Stock recommendation result"""
    action: Action
    confidence: int  # 0-100
    target_buy_price: Optional[float]
    target_sell_price: Optional[float]
    stop_loss: Optional[float]
    fair_value: Optional[float]
    reasoning: list


class RecommendationEngine:
    """Combines analysis results to generate buy/sell recommendations"""

    def __init__(
        self,
        sentiment_weight: float = 0.25,
        technical_weight: float = 0.40,
        fundamental_weight: float = 0.35
    ):
        """Initialize engine with analysis weights

        Args:
            sentiment_weight: Weight for sentiment analysis (0-1)
            technical_weight: Weight for technical analysis (0-1)
            fundamental_weight: Weight for fundamental analysis (0-1)

        Note: Weights are normalized if they don't sum to 1
        """
        total = sentiment_weight + technical_weight + fundamental_weight
        self.sentiment_weight = sentiment_weight / total
        self.technical_weight = technical_weight / total
        self.fundamental_weight = fundamental_weight / total

    def generate_recommendation(
        self,
        technical_analysis: Optional[dict] = None,
        fundamental_analysis: Optional[dict] = None,
        sentiment_analysis: Optional[dict] = None,
        transcript_analysis: Optional[dict] = None,
        current_price: Optional[float] = None,
        fmp_data: Optional[dict] = None
    ) -> Recommendation:
        """Generate a recommendation based on all analyses

        Args:
            technical_analysis: Results from TechnicalAnalyzer
            fundamental_analysis: Results from FundamentalAnalyzer
            sentiment_analysis: Results from SentimentAnalyzer
            transcript_analysis: Results from TranscriptAnalyzer
            current_price: Current stock price
            fmp_data: Optional FMP data for enhanced price targets

        Returns:
            Recommendation object with action, confidence, and targets
        """
        scores = []
        weights = []
        reasoning = []

        # Technical score
        if technical_analysis:
            tech_score = technical_analysis.get("score", 0)
            scores.append(tech_score)
            weights.append(self.technical_weight)

            trend = technical_analysis.get("trend", "neutral")
            rsi = technical_analysis.get("indicators", {}).get("rsi")
            reasoning.append(f"Technical: {trend} trend" + (f", RSI={rsi:.1f}" if rsi else ""))

        # Fundamental score
        if fundamental_analysis:
            fund_score = fundamental_analysis.get("score", 0)
            scores.append(fund_score)
            weights.append(self.fundamental_weight)

            metrics = fundamental_analysis.get("metrics", {})
            pe = metrics.get("pe_ratio")
            dcf = metrics.get("dcf_value")
            fmp_rating = metrics.get("fmp_rating")

            reason_parts = []
            if pe:
                reason_parts.append(f"P/E={pe:.1f}")
            if dcf and current_price:
                upside = ((dcf - current_price) / current_price) * 100
                reason_parts.append(f"DCF={dcf:.2f} ({upside:+.1f}%)")
            if fmp_rating:
                reason_parts.append(f"Rating={fmp_rating}")

            if reason_parts:
                reasoning.append(f"Fundamental: {', '.join(reason_parts)}")

        # Sentiment score
        if sentiment_analysis:
            sent_score = sentiment_analysis.get("score", 0)
            scores.append(sent_score)
            weights.append(self.sentiment_weight)

            classification = sentiment_analysis.get("classification", "neutral")
            article_count = sentiment_analysis.get("articles_analyzed", 0)
            reasoning.append(f"Sentiment: {classification} ({article_count} articles)")

        # Transcript analysis score
        if transcript_analysis and transcript_analysis.get("sentiment"):
            trans_sentiment = transcript_analysis.get("sentiment", {})
            trans_score = transcript_analysis.get("score", 0)

            # Transcript shares weight with sentiment (adds context to news sentiment)
            scores.append(trans_score)
            weights.append(self.sentiment_weight * 0.4)  # 40% of sentiment weight

            outlook = transcript_analysis.get("outlook", "neutral")
            confidence = trans_sentiment.get("confidence", "low")
            reasoning.append(f"Earnings Call: {outlook} outlook ({confidence} confidence)")

        # Calculate weighted score
        if scores:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]  # Normalize
            combined_score = sum(s * w for s, w in zip(scores, weights))
        else:
            combined_score = 0

        # Determine action
        if combined_score >= 50:
            action = Action.STRONG_BUY
        elif combined_score >= 25:
            action = Action.BUY
        elif combined_score <= -50:
            action = Action.STRONG_SELL
        elif combined_score <= -25:
            action = Action.SELL
        else:
            action = Action.HOLD

        # Calculate confidence (based on agreement between analyses)
        if len(scores) >= 2:
            # Check if analyses agree
            positive_count = sum(1 for s in scores if s > 10)
            negative_count = sum(1 for s in scores if s < -10)

            if positive_count == len(scores) or negative_count == len(scores):
                confidence = min(90, abs(combined_score) + 30)  # Strong agreement
            elif positive_count > 0 and negative_count > 0:
                confidence = max(30, abs(combined_score))  # Mixed signals
            else:
                confidence = min(70, abs(combined_score) + 10)
        else:
            confidence = min(50, abs(combined_score))  # Less confident with fewer inputs

        confidence = max(0, min(100, int(confidence)))

        # Calculate price targets
        target_buy_price = None
        target_sell_price = None
        stop_loss = None
        fair_value = None

        # Extract FMP-based targets if available
        analyst_target = None
        dcf_value = None

        if fundamental_analysis:
            metrics = fundamental_analysis.get("metrics", {})
            analyst_target = metrics.get("analyst_target_avg")
            dcf_value = metrics.get("dcf_value")

        # Calculate fair value (weighted average of DCF and analyst target)
        if dcf_value and analyst_target:
            fair_value = round((dcf_value * 0.5 + analyst_target * 0.5), 2)
        elif dcf_value:
            fair_value = round(dcf_value, 2)
        elif analyst_target:
            fair_value = round(analyst_target, 2)

        if current_price:
            # Technical-based support/resistance
            support = None
            resistance = None
            atr = None

            if technical_analysis:
                support = technical_analysis.get("support")
                resistance = technical_analysis.get("resistance")
                atr = technical_analysis.get("indicators", {}).get("atr")

            # Target buy price: use support level or discount from fair value
            if support:
                target_buy_price = round(support * 1.02, 2)
            elif fair_value and fair_value < current_price:
                # If overvalued, suggest buying at fair value
                target_buy_price = round(fair_value * 0.95, 2)
            elif current_price:
                # Default: 5% below current price
                target_buy_price = round(current_price * 0.95, 2)

            # Target sell price: use resistance, fair value, or analyst target
            if fair_value and fair_value > current_price:
                target_sell_price = round(fair_value, 2)
            elif resistance:
                target_sell_price = round(resistance * 0.98, 2)
            elif analyst_target and analyst_target > current_price:
                target_sell_price = round(analyst_target, 2)

            # Stop loss: below support (using ATR if available)
            if atr:
                stop_loss = round(current_price - (atr * 2), 2)
            elif support:
                stop_loss = round(support * 0.95, 2)
            else:
                # Default: 10% below current price
                stop_loss = round(current_price * 0.90, 2)

        # Add analyst/DCF reasoning if available
        if fundamental_analysis:
            metrics = fundamental_analysis.get("metrics", {})
            analyst_upside = metrics.get("analyst_upside")
            analyst_count = metrics.get("analyst_count")

            if analyst_upside is not None and analyst_count:
                reasoning.append(
                    f"Analysts: {analyst_upside:+.1f}% upside ({analyst_count} analysts)"
                )

            insider_buy = metrics.get("insider_buy_count", 0)
            insider_sell = metrics.get("insider_sell_count", 0)
            if insider_buy or insider_sell:
                reasoning.append(
                    f"Insiders: {insider_buy} buys, {insider_sell} sells (recent)"
                )

        return Recommendation(
            action=action,
            confidence=confidence,
            target_buy_price=target_buy_price,
            target_sell_price=target_sell_price,
            stop_loss=stop_loss,
            fair_value=fair_value,
            reasoning=reasoning,
        )

    def to_dict(self, recommendation: Recommendation) -> dict:
        """Convert recommendation to JSON-serializable dict"""
        return {
            "action": recommendation.action.value,
            "confidence": recommendation.confidence,
            "target_buy_price": recommendation.target_buy_price,
            "target_sell_price": recommendation.target_sell_price,
            "stop_loss": recommendation.stop_loss,
            "fair_value": recommendation.fair_value,
            "reasoning": recommendation.reasoning,
        }
