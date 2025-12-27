#!/usr/bin/env python3
"""Command-line interface for StockML"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

# Load .env file from current directory
load_dotenv()

from stockml import StockAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Stock Analysis Tool - Analyze stocks using technical, fundamental, and sentiment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py AAPL                    # Analyze Apple stock
  python cli.py AAPL GOOGL MSFT         # Compare multiple stocks
  python cli.py AAPL --output report.json   # Save to file
  python cli.py AAPL --period 6mo       # Use 6 months of history
  python cli.py AAPL --quick            # Just show recommendation
  python cli.py AAPL --summary          # Show human-readable summary

Environment variables:
  NEWS_API_KEY    Your NewsAPI.org API key for sentiment analysis
  FMP_API_KEY     Your Financial Modeling Prep API key for enhanced data
  OPENAI_API_KEY  Your OpenAI API key for AI-powered transcript summaries (optional)
        """
    )

    parser.add_argument(
        "tickers",
        nargs="+",
        help="Stock ticker symbol(s) to analyze (e.g., AAPL GOOGL)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path for JSON report"
    )

    parser.add_argument(
        "-p", "--period",
        default="1y",
        choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        help="Historical data period (default: 1y)"
    )

    parser.add_argument(
        "--news-days",
        type=int,
        default=7,
        help="Number of days of news to analyze (default: 7)"
    )

    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Skip news/sentiment analysis"
    )

    parser.add_argument(
        "--no-fmp",
        action="store_true",
        help="Skip FMP enhanced data (DCF, analyst ratings, insider trading)"
    )

    parser.add_argument(
        "--no-transcripts",
        action="store_true",
        help="Skip earnings transcript analysis"
    )

    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Show only the recommendation (BUY/SELL/HOLD)"
    )

    parser.add_argument(
        "-s", "--summary",
        action="store_true",
        help="Show human-readable summary instead of JSON"
    )

    parser.add_argument(
        "--technical-weight",
        type=float,
        default=0.40,
        help="Weight for technical analysis (default: 0.40)"
    )

    parser.add_argument(
        "--fundamental-weight",
        type=float,
        default=0.35,
        help="Weight for fundamental analysis (default: 0.35)"
    )

    parser.add_argument(
        "--sentiment-weight",
        type=float,
        default=0.25,
        help="Weight for sentiment analysis (default: 0.25)"
    )

    parser.add_argument(
        "--news-api-key",
        help="NewsAPI.org API key (or set NEWS_API_KEY env var)"
    )

    parser.add_argument(
        "--fmp-api-key",
        help="Financial Modeling Prep API key (or set FMP_API_KEY env var)"
    )

    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key for AI-powered transcript summaries (or set OPENAI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Get API keys
    news_api_key = args.news_api_key or os.environ.get("NEWS_API_KEY")
    fmp_api_key = args.fmp_api_key or os.environ.get("FMP_API_KEY")
    openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")

    # Show warnings for missing keys
    warnings = []
    if not news_api_key and not args.no_news:
        warnings.append("No NEWS_API_KEY set - will use FMP news if available")
    if not fmp_api_key and not args.no_fmp:
        warnings.append("No FMP_API_KEY set - DCF, analyst ratings, insider data, transcripts unavailable")
    if not openai_api_key and not args.no_transcripts:
        warnings.append("No OPENAI_API_KEY set - using extractive summaries for transcripts")

    if warnings:
        for w in warnings:
            print(f"Note: {w}", file=sys.stderr)
        print(file=sys.stderr)

    # Initialize analyzer
    analyzer = StockAnalyzer(
        news_api_key=news_api_key,
        fmp_api_key=fmp_api_key,
        openai_api_key=openai_api_key,
        technical_weight=args.technical_weight,
        fundamental_weight=args.fundamental_weight,
        sentiment_weight=args.sentiment_weight,
    )

    # Analyze stocks
    results = []

    for ticker in args.tickers:
        try:
            print(f"Analyzing {ticker}...", file=sys.stderr)

            report = analyzer.analyze(
                ticker,
                period=args.period,
                news_days=args.news_days,
                include_news=not args.no_news,
                include_fmp=not args.no_fmp,
                include_transcripts=not args.no_transcripts,
            )
            results.append(report)

            if args.quick:
                rec = report["recommendation"]
                fair_value = rec.get("fair_value")
                fv_str = f", fair value: ${fair_value:.2f}" if fair_value else ""
                print(f"{ticker}: {rec['action']} (confidence: {rec['confidence']}%{fv_str})")

        except Exception as e:
            print(f"Error analyzing {ticker}: {e}", file=sys.stderr)

    if args.quick:
        return

    # Output results
    if args.summary:
        for report in results:
            print_summary(report)
        return

    if len(results) == 1:
        output = results[0]
    else:
        output = {"stocks": results}

    json_output = json.dumps(output, indent=2, default=str)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print(json_output)


def print_summary(report: dict) -> None:
    """Print a human-readable summary of the report"""
    ticker = report["ticker"]
    price = report["current_price"]
    rec = report["recommendation"]

    print(f"\n{'='*60}")
    print(f"  {ticker} - ${price:.2f}")
    print(f"{'='*60}")
    print(f"  Recommendation: {rec['action']}")
    print(f"  Confidence: {rec['confidence']}%")

    if rec.get("fair_value"):
        upside = ((rec["fair_value"] - price) / price) * 100
        print(f"  Fair Value: ${rec['fair_value']:.2f} ({upside:+.1f}%)")

    if rec.get("target_buy_price"):
        print(f"  Target Buy Price: ${rec['target_buy_price']:.2f}")
    if rec.get("target_sell_price"):
        print(f"  Target Sell Price: ${rec['target_sell_price']:.2f}")
    if rec.get("stop_loss"):
        print(f"  Stop Loss: ${rec['stop_loss']:.2f}")

    # FMP data if available
    fmp = report.get("fmp_data", {})
    if fmp:
        print(f"\n  FMP Data:")
        if fmp.get("dcf_value"):
            print(f"    DCF Value: ${fmp['dcf_value']:.2f}")
        if fmp.get("analyst_target"):
            print(f"    Analyst Target: ${fmp['analyst_target']:.2f}")
        if fmp.get("fmp_rating"):
            print(f"    FMP Rating: {fmp['fmp_rating']} ({fmp.get('fmp_recommendation', 'N/A')})")

    print(f"\n  Reasoning:")
    for reason in rec.get("reasoning", []):
        print(f"    - {reason}")

    # Technical indicators
    tech = report.get("analysis", {}).get("technical", {})
    if tech:
        indicators = tech.get("indicators", {})
        print(f"\n  Technical Indicators:")
        if indicators.get("rsi"):
            print(f"    RSI: {indicators['rsi']:.1f}")
        if tech.get("trend"):
            print(f"    Trend: {tech['trend']}")
        if tech.get("support") and tech.get("resistance"):
            print(f"    Support: ${tech['support']:.2f} | Resistance: ${tech['resistance']:.2f}")

    # Fundamental metrics
    fund = report.get("analysis", {}).get("fundamental", {})
    if fund:
        metrics = fund.get("metrics", {})
        print(f"\n  Fundamental Metrics:")
        if metrics.get("pe_ratio"):
            print(f"    P/E Ratio: {metrics['pe_ratio']:.1f}")
        if metrics.get("dividend_yield"):
            div_yield = metrics['dividend_yield']
            div_yield_display = div_yield * 100 if div_yield < 0.2 else div_yield
            print(f"    Dividend Yield: {div_yield_display:.2f}%")
        if metrics.get("insider_buy_count") or metrics.get("insider_sell_count"):
            print(f"    Insider Activity: {metrics.get('insider_buy_count', 0)} buys, {metrics.get('insider_sell_count', 0)} sells")

    # Transcript analysis
    transcript = report.get("analysis", {}).get("transcript", {})
    if transcript:
        print(f"\n  Earnings Transcript:")
        if transcript.get("quarter") and transcript.get("year"):
            print(f"    Latest Call: Q{transcript['quarter']} {transcript['year']}")
        if transcript.get("outlook"):
            print(f"    Outlook: {transcript['outlook']}")
        if transcript.get("sentiment_score") is not None:
            print(f"    Sentiment: {transcript['sentiment_score']:.2f} ({transcript.get('confidence', 'N/A')} confidence)")
        if transcript.get("key_points"):
            print(f"    Key Points:")
            for point in transcript["key_points"][:3]:
                # Truncate long points
                point_text = point[:100] + "..." if len(point) > 100 else point
                print(f"      - {point_text}")

    print()


if __name__ == "__main__":
    main()
