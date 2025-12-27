"""Earnings transcript analysis"""

import re
from typing import Optional, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TranscriptAnalyzer:
    """Analyze earnings call transcripts for sentiment, metrics, and key points"""

    # Financial keywords for enhanced sentiment analysis
    BULLISH_KEYWORDS = [
        "growth", "growing", "increase", "increased", "increasing",
        "strong", "stronger", "strength", "momentum", "accelerating",
        "exceed", "exceeded", "exceeding", "beat", "beating",
        "record", "highest", "best", "outperform", "outperforming",
        "optimistic", "confident", "confident", "excited", "exciting",
        "opportunity", "opportunities", "upside", "tailwind", "tailwinds",
        "expand", "expanding", "expansion", "improve", "improving",
        "robust", "solid", "healthy", "resilient", "innovation",
    ]

    BEARISH_KEYWORDS = [
        "decline", "declining", "decrease", "decreased", "decreasing",
        "weak", "weaker", "weakness", "slowdown", "slowing",
        "miss", "missed", "missing", "below", "under",
        "challenge", "challenges", "challenging", "difficult", "headwind",
        "headwinds", "concern", "concerned", "concerns", "uncertain",
        "uncertainty", "risk", "risks", "pressure", "pressures",
        "downturn", "recession", "contraction", "cautious", "conservative",
        "disappointing", "disappointed", "restructuring", "layoff", "layoffs",
    ]

    # Patterns for extracting financial guidance
    REVENUE_PATTERNS = [
        r"revenue[s]?\s+(?:guidance|outlook|expectation)[s]?\s+(?:of|is|are)?\s*\$?([\d.,]+)\s*(billion|million|B|M)?",
        r"expect[s]?\s+revenue[s]?\s+(?:of|to be|around)?\s*\$?([\d.,]+)\s*(billion|million|B|M)?",
        r"revenue[s]?\s+(?:of|around|approximately)?\s*\$?([\d.,]+)\s*(billion|million|B|M)?",
    ]

    EPS_PATTERNS = [
        r"(?:eps|earnings per share)\s+(?:guidance|outlook|expectation)[s]?\s+(?:of|is|are)?\s*\$?([\d.]+)",
        r"expect[s]?\s+(?:eps|earnings per share)\s+(?:of|to be|around)?\s*\$?([\d.]+)",
        r"(?:eps|earnings per share)\s+(?:of|around|approximately)?\s*\$?([\d.]+)",
    ]

    GROWTH_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+(?:growth|increase|improvement)",
        r"grow(?:th|ing)?\s+(?:of|by|at)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent)",
        r"(?:year-over-year|yoy|y/y)\s+(?:growth|increase)\s+(?:of)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent)?",
    ]

    MARGIN_PATTERNS = [
        r"(?:gross|operating|net|profit)\s+margin[s]?\s+(?:of|at|around)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent)?",
        r"margin[s]?\s+(?:expansion|improvement|increase)\s+(?:of|by)?\s*(\d+(?:\.\d+)?)\s*(?:basis points|bps|%|percent)?",
    ]

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize TranscriptAnalyzer

        Args:
            openai_api_key: Optional OpenAI API key for AI-powered summaries.
                           If not provided, uses extractive summarization.
        """
        self.openai_api_key = openai_api_key
        self._vader = SentimentIntensityAnalyzer()
        self._openai_client = None

    def _get_openai_client(self):
        """Lazily initialize OpenAI client"""
        if self._openai_client is None and self.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except ImportError:
                pass
        return self._openai_client

    def analyze_sentiment(self, transcript: str) -> dict:
        """Analyze transcript sentiment using VADER + financial keyword analysis

        Args:
            transcript: Full transcript text

        Returns:
            Dict with sentiment scores and classification
        """
        # VADER sentiment on full text (sample if too long)
        text_sample = transcript[:50000] if len(transcript) > 50000 else transcript
        vader_scores = self._vader.polarity_scores(text_sample)

        # Count financial keywords
        text_lower = transcript.lower()

        bullish_count = sum(
            len(re.findall(r'\b' + kw + r'\b', text_lower))
            for kw in self.BULLISH_KEYWORDS
        )

        bearish_count = sum(
            len(re.findall(r'\b' + kw + r'\b', text_lower))
            for kw in self.BEARISH_KEYWORDS
        )

        # Calculate keyword ratio
        total_keywords = bullish_count + bearish_count
        if total_keywords > 0:
            keyword_sentiment = (bullish_count - bearish_count) / total_keywords
        else:
            keyword_sentiment = 0

        # Combine VADER and keyword sentiment (60% VADER, 40% keywords)
        combined_sentiment = (vader_scores["compound"] * 0.6) + (keyword_sentiment * 0.4)

        # Determine confidence level
        if abs(combined_sentiment) > 0.3:
            confidence = "high"
        elif abs(combined_sentiment) > 0.1:
            confidence = "medium"
        else:
            confidence = "low"

        # Classification
        if combined_sentiment >= 0.1:
            classification = "bullish"
        elif combined_sentiment <= -0.1:
            classification = "bearish"
        else:
            classification = "neutral"

        return {
            "sentiment_score": round(combined_sentiment, 3),
            "vader_compound": vader_scores["compound"],
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "keyword_sentiment": round(keyword_sentiment, 3),
            "confidence": confidence,
            "classification": classification,
        }

    def extract_metrics(self, transcript: str) -> dict:
        """Extract financial metrics and guidance from transcript

        Args:
            transcript: Full transcript text

        Returns:
            Dict with extracted metrics
        """
        text = transcript.lower()
        metrics = {
            "revenue_mentions": [],
            "eps_mentions": [],
            "growth_mentions": [],
            "margin_mentions": [],
        }

        # Extract revenue guidance
        for pattern in self.REVENUE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    value = match[0]
                    unit = match[1] if len(match) > 1 else ""
                else:
                    value = match
                    unit = ""
                metrics["revenue_mentions"].append(f"${value} {unit}".strip())

        # Extract EPS guidance
        for pattern in self.EPS_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match if isinstance(match, str) else match[0]
                metrics["eps_mentions"].append(f"${value}")

        # Extract growth percentages
        for pattern in self.GROWTH_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match if isinstance(match, str) else match[0]
                metrics["growth_mentions"].append(f"{value}%")

        # Extract margin data
        for pattern in self.MARGIN_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match if isinstance(match, str) else match[0]
                metrics["margin_mentions"].append(f"{value}%")

        # Deduplicate
        metrics["revenue_mentions"] = list(set(metrics["revenue_mentions"]))[:5]
        metrics["eps_mentions"] = list(set(metrics["eps_mentions"]))[:5]
        metrics["growth_mentions"] = list(set(metrics["growth_mentions"]))[:10]
        metrics["margin_mentions"] = list(set(metrics["margin_mentions"]))[:5]

        return metrics

    def _extract_key_sentences(self, transcript: str, max_sentences: int = 5) -> List[str]:
        """Extract key sentences using simple heuristics"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 50]

        # Score sentences by keyword presence
        scored = []
        important_phrases = [
            "guidance", "outlook", "expect", "forecast", "quarter",
            "revenue", "earnings", "growth", "margin", "cash flow",
            "year-over-year", "strong", "confident", "momentum",
        ]

        for sentence in sentences:
            score = sum(1 for phrase in important_phrases if phrase in sentence.lower())
            if score > 0:
                scored.append((score, sentence))

        # Sort by score and return top sentences
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1][:500] for s in scored[:max_sentences]]

    def generate_summary(
        self,
        transcript: str,
        company_name: str,
        use_ai: bool = True
    ) -> dict:
        """Generate transcript summary

        Args:
            transcript: Full transcript text
            company_name: Company name for context
            use_ai: Whether to use OpenAI for summary (if available)

        Returns:
            Dict with summary and key points
        """
        # Try AI summary first if enabled and available
        if use_ai:
            client = self._get_openai_client()
            if client:
                try:
                    # Truncate transcript to fit context window
                    max_chars = 100000  # ~25k tokens
                    truncated = transcript[:max_chars]

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a financial analyst summarizing earnings calls. Be concise and focus on key metrics, guidance, and management outlook."
                            },
                            {
                                "role": "user",
                                "content": f"""Summarize this earnings call transcript for {company_name}.

Provide:
1. A 2-3 paragraph summary of key points
2. The overall outlook (bullish/neutral/bearish)
3. 3-5 key takeaways as bullet points

Transcript:
{truncated}"""
                            }
                        ],
                        max_tokens=1000,
                        temperature=0.3,
                    )

                    ai_response = response.choices[0].message.content

                    # Parse AI response
                    lines = ai_response.split('\n')
                    summary_lines = []
                    key_points = []
                    outlook = "neutral"

                    in_takeaways = False
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        if "outlook" in line.lower():
                            if "bullish" in line.lower():
                                outlook = "bullish"
                            elif "bearish" in line.lower():
                                outlook = "bearish"
                        elif line.startswith(("-", "•", "*")) or (line[0].isdigit() and line[1] in ".):"):
                            key_points.append(line.lstrip("-•* 0123456789.)"))
                            in_takeaways = True
                        elif not in_takeaways:
                            summary_lines.append(line)

                    return {
                        "summary": "\n\n".join(summary_lines) if summary_lines else ai_response,
                        "key_points": key_points[:5],
                        "outlook": outlook,
                        "method": "ai",
                    }

                except Exception as e:
                    # Fall through to extractive summary
                    pass

        # Fallback: extractive summary
        key_sentences = self._extract_key_sentences(transcript)
        sentiment = self.analyze_sentiment(transcript)

        return {
            "summary": " ".join(key_sentences) if key_sentences else "No summary available.",
            "key_points": key_sentences[:3],
            "outlook": sentiment["classification"],
            "method": "extractive",
        }

    def analyze(
        self,
        transcript: str,
        company_name: str,
        use_ai: bool = True
    ) -> dict:
        """Perform full transcript analysis

        Args:
            transcript: Full transcript text
            company_name: Company name
            use_ai: Whether to use AI for summary

        Returns:
            Complete analysis dict
        """
        if not transcript or len(transcript) < 100:
            return {
                "error": "Transcript too short or empty",
                "sentiment": None,
                "metrics": None,
                "summary": None,
            }

        sentiment = self.analyze_sentiment(transcript)
        metrics = self.extract_metrics(transcript)
        summary = self.generate_summary(transcript, company_name, use_ai=use_ai)

        # Calculate overall score (-100 to 100)
        score = int(sentiment["sentiment_score"] * 100)

        return {
            "sentiment": sentiment,
            "metrics": metrics,
            "summary": summary["summary"],
            "key_points": summary["key_points"],
            "outlook": summary["outlook"],
            "summary_method": summary["method"],
            "score": score,
        }
