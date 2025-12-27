# StockML

A comprehensive stock analysis framework combining technical, fundamental, and sentiment analysis with AI-powered insights.

## Features

- **Technical Analysis**: RSI, MACD, moving averages, support/resistance levels, trend detection
- **Fundamental Analysis**: Valuation metrics (P/E, P/B, PEG), profitability, financial health, dividends
- **News Sentiment Analysis**: Aggregates and analyzes recent news articles
- **Earnings Transcript Analysis**: Parses earnings calls for outlook and key points
- **AI Investment Narrative**: GPT-powered 2-5 year investment outlook synthesis
- **Peer Comparison**: Compare against sector peers with similar market cap
- **Interactive TUI**: Terminal-based interface with tabs, navigation, and live data
- **CLI Interface**: Quick analysis from the command line

## Installation

```bash
# Clone the repository
git clone https://github.com/ajokela/stockml.git
cd stockml

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## API Keys

StockML uses several APIs to fetch data. Create a `.env` file in the project root:

```bash
# Required for basic functionality
FMP_API_KEY=your_fmp_key_here

# Optional - enhances news sentiment analysis
NEWS_API_KEY=your_newsapi_key_here

# Optional - enables AI-powered investment narrative
OPENAI_API_KEY=your_openai_key_here
```

### Where to Get API Keys

| API | Purpose | Get Key |
|-----|---------|---------|
| **Financial Modeling Prep (FMP)** | Fundamentals, DCF, analyst ratings, peers, transcripts | [financialmodelingprep.com](https://site.financialmodelingprep.com/developer/docs) - Free tier: 250 requests/day |
| **NewsAPI** | News article fetching for sentiment | [newsapi.org](https://newsapi.org/register) - Free tier: 100 requests/day |
| **OpenAI** | AI-powered investment narrative | [platform.openai.com](https://platform.openai.com/api-keys) - Pay per use |

**Note**: Yahoo Finance data (price history, basic info) works without any API key.

## Usage

### Interactive TUI

```bash
# Launch the terminal UI
python tui.py AAPL

# Or without a ticker (add via 'a' key)
python tui.py
```

**Keyboard shortcuts:**
- `1-6` - Switch tabs (Overview, Technical, Fundamental, Sentiment, Transcript, News)
- `a` - Add/change stock
- `r` - Refresh data
- `c` - Toggle comparison mode
- `Enter` - Open news article in browser (in News tab)
- `q` - Quit

### CLI

```bash
# Quick analysis
python cli.py AAPL

# With options
python cli.py AAPL --period 2y --no-news
```

### Python API

```python
from stockml import StockAnalyzer

analyzer = StockAnalyzer(
    fmp_api_key="your_key",
    news_api_key="your_key",
    openai_api_key="your_key"
)

# Full analysis
report = analyzer.analyze("AAPL")
print(report["recommendation"]["action"])  # BUY, SELL, HOLD, etc.
print(report["investment_narrative"])      # AI-generated outlook

# Individual analyses
technical = analyzer.get_technical_analysis("AAPL")
fundamental = analyzer.get_fundamental_analysis("AAPL")
sentiment = analyzer.get_sentiment_analysis("AAPL")
```

## Project Structure

```
stockml/
├── stockml/
│   ├── analysis/
│   │   ├── technical.py      # Technical indicators
│   │   ├── fundamental.py    # Fundamental metrics
│   │   ├── sentiment.py      # News sentiment
│   │   └── transcript.py     # Earnings call analysis
│   ├── data/
│   │   ├── yahoo.py          # Yahoo Finance client
│   │   ├── fmp.py            # Financial Modeling Prep client
│   │   └── news.py           # NewsAPI client
│   ├── tui/
│   │   ├── app.py            # Main TUI application
│   │   ├── widgets/          # Custom Textual widgets
│   │   └── styles/           # TUI stylesheets
│   ├── analyzer.py           # Main StockAnalyzer class
│   ├── recommendation.py     # Recommendation engine
│   └── report.py             # Report generation
├── cli.py                    # CLI entry point
├── tui.py                    # TUI entry point
├── requirements.txt
└── .env                      # API keys (not committed)
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.
