"""Main StockML TUI Application"""

import os
import webbrowser
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Static,
    TabbedContent,
    TabPane,
    DataTable,
    Input,
    Button,
    LoadingIndicator,
    Label,
    ListView,
    ListItem,
    Markdown,
)
from textual.screen import ModalScreen

from .widgets.recommendation import RecommendationWidget
from .widgets.score_bar import ScoreBar
from .widgets.price_targets import PriceTargetsWidget
from .widgets.metrics_panel import MetricsPanel


class AddStockModal(ModalScreen):
    """Modal dialog to add a new stock"""

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="add-stock-dialog"):
            yield Label("Enter Stock Ticker:")
            yield Input(placeholder="e.g., AAPL", id="ticker-input")
            with Horizontal():
                yield Button("Add", variant="primary", id="add-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add-btn":
            ticker = self.query_one("#ticker-input", Input).value.strip().upper()
            if ticker:
                self.dismiss(ticker)
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        ticker = event.value.strip().upper()
        if ticker:
            self.dismiss(ticker)


class StockMLApp(App):
    """StockML Terminal User Interface"""

    CSS_PATH = Path(__file__).parent / "styles" / "app.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("a", "add_stock", "Add Stock"),
        Binding("c", "toggle_compare", "Compare"),
        Binding("1", "tab_1", "Overview", show=False),
        Binding("2", "tab_2", "Technical", show=False),
        Binding("3", "tab_3", "Fundamental", show=False),
        Binding("4", "tab_4", "Sentiment", show=False),
        Binding("5", "tab_5", "Transcript", show=False),
        Binding("6", "tab_6", "News", show=False),
    ]

    def __init__(
        self,
        ticker: str = None,
        analyzer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ticker = ticker
        self.analyzer = analyzer
        self.report: Optional[dict] = None
        self.stocks: list = []  # For comparison mode
        self.compare_mode = False
        self.news_urls: list = []  # Store URLs for news items

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-container"):
            # Stock header with ticker and price
            with Horizontal(id="stock-header"):
                yield Static("Loading...", id="stock-info")
                yield Static("", id="stock-price")

            # Main tabbed content
            with TabbedContent(id="main-tabs"):
                with TabPane("Overview", id="tab-overview"):
                    yield from self._compose_overview_tab()

                with TabPane("Technical", id="tab-technical"):
                    yield from self._compose_technical_tab()

                with TabPane("Fundamental", id="tab-fundamental"):
                    yield from self._compose_fundamental_tab()

                with TabPane("Sentiment", id="tab-sentiment"):
                    yield from self._compose_sentiment_tab()

                with TabPane("Transcript", id="tab-transcript"):
                    yield from self._compose_transcript_tab()

                with TabPane("News", id="tab-news"):
                    yield from self._compose_news_tab()

        yield Footer()

    def _compose_overview_tab(self) -> ComposeResult:
        with ScrollableContainer():
            with Horizontal(id="overview-top"):
                with Container(id="recommendation-container"):
                    yield RecommendationWidget(id="recommendation")
                with Container(id="price-targets-container"):
                    yield PriceTargetsWidget(id="price-targets")

            with Container(id="company-summary-container"):
                yield Static("[bold]About[/bold]", id="company-summary-title")
                yield Static("", id="company-summary")

            with Container(id="scores-container"):
                yield Static("[bold]Analysis Scores[/bold]", id="scores-title")
                yield ScoreBar("Technical", 0, "", id="score-technical")
                yield ScoreBar("Fundamental", 0, "", id="score-fundamental")
                yield ScoreBar("Sentiment", 0, "", id="score-sentiment")
                yield ScoreBar("Transcript", 0, "", id="score-transcript")

            with Container(id="reasoning-container"):
                yield Static("[bold]Reasoning[/bold]")
                yield Static("", id="reasoning-list")

            with Container(id="narrative-container"):
                yield Static("[bold]Investment Outlook (2-5 Year Horizon)[/bold]", id="narrative-title")
                yield Static("", id="investment-narrative")

            with Container(id="peers-container"):
                yield Static("[bold]Peer Comparison[/bold]", id="peers-title")
                yield DataTable(id="peers-table")

    def _compose_technical_tab(self) -> ComposeResult:
        with Vertical():
            with Horizontal():
                yield MetricsPanel(title="Trend & Levels", id="tech-trend")
                yield MetricsPanel(title="Key Indicators", id="tech-indicators")

            yield Static("[bold]Technical Signals[/bold]")
            yield DataTable(id="tech-signals-table")

    def _compose_fundamental_tab(self) -> ComposeResult:
        with ScrollableContainer():
            with Horizontal():
                yield MetricsPanel(title="Valuation", id="fund-valuation")
                yield MetricsPanel(title="Financial Health", id="fund-health")

            with Horizontal():
                yield MetricsPanel(title="Profitability", id="fund-profit")
                yield MetricsPanel(title="Growth", id="fund-growth")

            with Horizontal():
                yield MetricsPanel(title="Dividends", id="fund-dividends")
                yield MetricsPanel(title="Analyst Data", id="fund-analyst")

    def _compose_sentiment_tab(self) -> ComposeResult:
        with Vertical():
            yield MetricsPanel(title="Sentiment Analysis", id="sentiment-panel")
            yield ScoreBar("Overall", 0, "", id="sentiment-bar")

    def _compose_transcript_tab(self) -> ComposeResult:
        with ScrollableContainer():
            yield MetricsPanel(title="Earnings Call Info", id="transcript-info")
            yield Static("[bold]Key Points[/bold]")
            yield Static("", id="transcript-points")
            yield Static("[bold]Summary[/bold]")
            yield Static("", id="transcript-summary")

    def _compose_news_tab(self) -> ComposeResult:
        yield ListView(id="news-list")

    async def on_mount(self) -> None:
        """Load data when app starts"""
        if self.ticker:
            await self.load_stock(self.ticker)

    async def load_stock(self, ticker: str) -> None:
        """Load and analyze a stock"""
        self.ticker = ticker.upper()

        # Update header
        stock_info = self.query_one("#stock-info", Static)
        stock_info.update(f"Loading {self.ticker}...")

        # Run analysis in background thread (blocking I/O)
        self.run_worker(lambda: self._analyze_stock(ticker), exclusive=True, thread=True)

    def _analyze_stock(self, ticker: str) -> None:
        """Perform stock analysis (runs in worker thread)"""
        from dotenv import load_dotenv
        load_dotenv()

        if self.analyzer is None:
            from stockml import StockAnalyzer
            self.analyzer = StockAnalyzer(
                news_api_key=os.environ.get("NEWS_API_KEY"),
                fmp_api_key=os.environ.get("FMP_API_KEY"),
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
            )

        try:
            self.report = self.analyzer.analyze(ticker)
            self.call_from_thread(self._update_display)
        except Exception as e:
            self.call_from_thread(self._show_error, str(e))

    def _show_error(self, message: str) -> None:
        """Show error message"""
        stock_info = self.query_one("#stock-info", Static)
        stock_info.update(f"[red]Error: {message}[/red]")

    def _update_display(self) -> None:
        """Update all display components with report data"""
        if not self.report:
            return

        # Update stock header
        ticker = self.report.get("ticker", self.ticker)
        price = self.report.get("current_price", 0)
        company = self.report.get("analysis", {}).get("fundamental", {}).get("company", {})
        company_name = company.get("name", ticker)

        stock_info = self.query_one("#stock-info", Static)
        stock_info.update(f"[bold]{ticker}[/bold] - {company_name}")

        stock_price = self.query_one("#stock-price", Static)
        stock_price.update(f"[bold]${price:.2f}[/bold]")

        self._update_overview()
        self._update_technical()
        self._update_fundamental()
        self._update_sentiment()
        self._update_transcript()
        self._update_news()

    def _update_overview(self) -> None:
        """Update overview tab"""
        rec = self.report.get("recommendation", {})
        price = self.report.get("current_price", 0)

        # Recommendation widget
        rec_widget = self.query_one("#recommendation", RecommendationWidget)
        rec_widget.update_recommendation(
            rec.get("action", "HOLD"),
            rec.get("confidence", 50)
        )

        # Price targets
        targets = self.query_one("#price-targets", PriceTargetsWidget)
        targets.update_targets(
            current_price=price,
            fair_value=rec.get("fair_value"),
            target_buy=rec.get("target_buy_price"),
            target_sell=rec.get("target_sell_price"),
            stop_loss=rec.get("stop_loss"),
        )

        # Company summary
        analysis = self.report.get("analysis", {})
        company = analysis.get("fundamental", {}).get("company", {})
        summary = company.get("summary", "")
        summary_widget = self.query_one("#company-summary", Static)
        if summary:
            # Truncate long summaries
            summary_clean = " ".join(summary.split())
            summary_truncated = summary_clean[:400] + "..." if len(summary_clean) > 400 else summary_clean
            # Add sector/industry context
            sector = company.get("sector", "")
            industry = company.get("industry", "")
            context = f"[dim]{sector} · {industry}[/dim]\n\n" if sector and industry else ""
            summary_widget.update(f"{context}{summary_truncated}")
        else:
            summary_widget.update("[dim]No company description available[/dim]")

        # Analysis scores

        tech = analysis.get("technical", {})
        if tech:
            score_tech = self.query_one("#score-technical", ScoreBar)
            score_tech.update_score(tech.get("score", 0), tech.get("trend", ""))

        fund = analysis.get("fundamental", {})
        if fund:
            score_fund = self.query_one("#score-fundamental", ScoreBar)
            score_fund.update_score(fund.get("score", 0), "")

        sent = analysis.get("sentiment", {})
        if sent:
            score_sent = self.query_one("#score-sentiment", ScoreBar)
            score_sent.update_score(sent.get("score", 0), sent.get("classification", ""))

        trans = analysis.get("transcript", {})
        if trans:
            score_trans = self.query_one("#score-transcript", ScoreBar)
            score_trans.update_score(trans.get("score", 0), trans.get("outlook", ""))

        # Reasoning
        reasoning = rec.get("reasoning", [])
        reasoning_text = "\n".join(f"  * {r}" for r in reasoning)
        reasoning_widget = self.query_one("#reasoning-list", Static)
        reasoning_widget.update(reasoning_text if reasoning else "No reasoning available")

        # Investment narrative
        narrative = self.report.get("investment_narrative", "")
        narrative_widget = self.query_one("#investment-narrative", Static)
        if narrative:
            narrative_widget.update(narrative)
        else:
            narrative_widget.update("[dim]Investment narrative requires OpenAI API key[/dim]")

        # Peer comparison
        self._update_peers()

    def _update_peers(self) -> None:
        """Update peer comparison table"""
        peers_table = self.query_one("#peers-table", DataTable)
        peers_table.clear(columns=True)

        fmp_data = self.report.get("fmp_data", {})
        peers = fmp_data.get("peers", [])

        if not peers:
            peers_table.add_column("Status")
            peers_table.add_row("[dim]No peer data available (requires FMP API)[/dim]")
            return

        # Add columns
        peers_table.add_columns("Ticker", "P/E", "P/B", "Market Cap", "Div Yield")

        # Add current stock first for comparison
        fund_metrics = self.report.get("analysis", {}).get("fundamental", {}).get("metrics", {})
        current_ticker = self.report.get("ticker", "")
        current_pe = fund_metrics.get("pe_ratio")
        current_pb = fund_metrics.get("price_to_book")
        current_mcap = fund_metrics.get("market_cap")
        current_div = fund_metrics.get("dividend_yield")

        # Format current stock row
        pe_str = f"{current_pe:.2f}" if current_pe else "--"
        pb_str = f"{current_pb:.2f}" if current_pb else "--"
        mcap_str = self._format_market_cap(current_mcap) if current_mcap else "--"
        div_str = f"{current_div * 100:.2f}%" if current_div and current_div < 0.2 else (f"{current_div:.2f}%" if current_div else "--")

        peers_table.add_row(
            f"[bold]{current_ticker}[/bold] *",
            pe_str,
            pb_str,
            mcap_str,
            div_str
        )

        # Add peer rows
        for peer in peers:
            ticker = peer.get("ticker", "")
            pe = peer.get("pe")
            pb = peer.get("pb")
            mcap = peer.get("market_cap")
            div_yield = peer.get("div_yield")

            pe_str = f"{pe:.2f}" if pe else "--"
            pb_str = f"{pb:.2f}" if pb else "--"
            mcap_str = self._format_market_cap(mcap) if mcap else "--"
            div_str = f"{div_yield:.2f}%" if div_yield else "--"

            peers_table.add_row(ticker, pe_str, pb_str, mcap_str, div_str)

    def _format_market_cap(self, value: float) -> str:
        """Format market cap value"""
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        else:
            return f"${value:,.0f}"

    def _update_technical(self) -> None:
        """Update technical tab"""
        tech = self.report.get("analysis", {}).get("technical", {})
        if not tech:
            return

        indicators = tech.get("indicators", {})

        # Trend & Levels
        trend_panel = self.query_one("#tech-trend", MetricsPanel)
        trend_panel.update_metrics({
            "Trend": tech.get("trend", "--"),
            "Score": (tech.get("score", 0), "int"),
            "Support": (tech.get("support"), "currency"),
            "Resistance": (tech.get("resistance"), "currency"),
        })

        # Key Indicators
        indicators_panel = self.query_one("#tech-indicators", MetricsPanel)
        indicators_panel.update_metrics({
            "RSI": (indicators.get("rsi"), "number"),
            "MACD": (indicators.get("macd"), "number"),
            "MACD Signal": (indicators.get("macd_signal"), "number"),
            "ATR": (indicators.get("atr"), "number"),
            "SMA 20": (indicators.get("sma_20"), "currency"),
            "SMA 50": (indicators.get("sma_50"), "currency"),
            "SMA 200": (indicators.get("sma_200"), "currency"),
        })

        # Signals table
        signals = tech.get("signals", [])
        table = self.query_one("#tech-signals-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Type", "Signal", "Strength")
        for sig in signals:
            strength = sig.get("strength", "")
            if "buy" in strength.lower():
                strength = f"[green]{strength}[/green]"
            elif "sell" in strength.lower():
                strength = f"[red]{strength}[/red]"
            table.add_row(
                sig.get("type", ""),
                sig.get("signal", ""),
                strength
            )

    def _update_fundamental(self) -> None:
        """Update fundamental tab"""
        fund = self.report.get("analysis", {}).get("fundamental", {})
        if not fund:
            return

        metrics = fund.get("metrics", {})

        # Valuation
        val_panel = self.query_one("#fund-valuation", MetricsPanel)
        val_panel.update_metrics({
            "P/E Ratio": (metrics.get("pe_ratio"), "number"),
            "PEG Ratio": (metrics.get("peg_ratio"), "number"),
            "P/B Ratio": (metrics.get("price_to_book"), "number"),
            "Market Cap": (metrics.get("market_cap"), "currency_large"),
            "DCF Value": (metrics.get("dcf_value"), "currency"),
        })

        # Financial Health
        health_panel = self.query_one("#fund-health", MetricsPanel)
        health_panel.update_metrics({
            "Debt/Equity": (metrics.get("debt_to_equity"), "number"),
            "Current Ratio": (metrics.get("current_ratio"), "number"),
            "Quick Ratio": (metrics.get("quick_ratio"), "number"),
        })

        # Profitability
        profit_panel = self.query_one("#fund-profit", MetricsPanel)
        profit_panel.update_metrics({
            "Profit Margin": (metrics.get("profit_margin") * 100, "percent") if metrics.get("profit_margin") else ("--", None),
            "Operating Margin": (metrics.get("operating_margin") * 100, "percent") if metrics.get("operating_margin") else ("--", None),
            "ROE": (metrics.get("return_on_equity") * 100, "percent") if metrics.get("return_on_equity") else ("--", None),
            "ROA": (metrics.get("return_on_assets") * 100, "percent") if metrics.get("return_on_assets") else ("--", None),
        })

        # Growth
        growth_panel = self.query_one("#fund-growth", MetricsPanel)
        growth_panel.update_metrics({
            "Revenue Growth": (metrics.get("revenue_growth") * 100, "percent") if metrics.get("revenue_growth") else ("--", None),
            "Earnings Growth": (metrics.get("earnings_growth") * 100, "percent") if metrics.get("earnings_growth") else ("--", None),
        })

        # Dividends
        div_panel = self.query_one("#fund-dividends", MetricsPanel)
        if metrics.get("has_dividends"):
            # Dividend yield from Yahoo can be inconsistent - sometimes decimal, sometimes already percentage
            # If value > 0.2 (20%), it's likely already in percentage form, don't multiply
            div_yield = metrics.get("dividend_yield", 0)
            div_yield_display = div_yield * 100 if div_yield < 0.2 else div_yield
            div_panel.update_metrics({
                "Dividend Yield": (div_yield_display, "percent"),
                "Dividend Rate": (metrics.get("dividend_rate"), "currency"),
                "Payout Ratio": (metrics.get("payout_ratio") * 100, "percent") if metrics.get("payout_ratio") else ("--", None),
            })
        else:
            div_panel.update_metrics({"Dividends": "None"})

        # Analyst Data
        analyst_panel = self.query_one("#fund-analyst", MetricsPanel)
        analyst_panel.update_metrics({
            "Analyst Target": (metrics.get("analyst_target_avg"), "currency"),
            "Analyst Count": (metrics.get("analyst_count"), "int"),
            "Upgrades": (metrics.get("recent_upgrades", 0), "int"),
            "Downgrades": (metrics.get("recent_downgrades", 0), "int"),
            "Insider Buys": (metrics.get("insider_buy_count", 0), "int"),
            "Insider Sells": (metrics.get("insider_sell_count", 0), "int"),
        })

    def _update_sentiment(self) -> None:
        """Update sentiment tab"""
        sent = self.report.get("analysis", {}).get("sentiment", {})
        if not sent:
            panel = self.query_one("#sentiment-panel", MetricsPanel)
            panel.update_metrics({"Status": "No sentiment data available"})
            return

        panel = self.query_one("#sentiment-panel", MetricsPanel)
        panel.update_metrics({
            "Classification": sent.get("classification", "--"),
            "Articles Analyzed": (sent.get("articles_analyzed", 0), "int"),
            "Positive": (sent.get("positive_count", 0), "int"),
            "Negative": (sent.get("negative_count", 0), "int"),
            "Neutral": (sent.get("neutral_count", 0), "int"),
        })

        bar = self.query_one("#sentiment-bar", ScoreBar)
        bar.update_score(sent.get("score", 0), sent.get("classification", ""))

    def _update_transcript(self) -> None:
        """Update transcript tab"""
        trans = self.report.get("analysis", {}).get("transcript", {})

        info_panel = self.query_one("#transcript-info", MetricsPanel)
        if not trans:
            info_panel.update_metrics({"Status": "No transcript available"})
            self.query_one("#transcript-points", Static).update("")
            self.query_one("#transcript-summary", Static).update("")
            return

        info_panel.update_metrics({
            "Quarter": f"Q{trans.get('quarter', '?')} {trans.get('year', '?')}",
            "Outlook": trans.get("outlook", "--"),
            "Confidence": trans.get("confidence", "--"),
            "Sentiment Score": (trans.get("sentiment_score"), "number"),
        })

        # Key points
        points = trans.get("key_points", [])
        points_text = "\n".join(f"  * {p[:100]}..." if len(p) > 100 else f"  * {p}" for p in points)
        self.query_one("#transcript-points", Static).update(points_text or "No key points")

        # Summary
        summary = trans.get("summary", "No summary available")
        self.query_one("#transcript-summary", Static).update(summary[:500] if summary else "No summary")

    def _update_news(self) -> None:
        """Update news tab"""
        news = self.report.get("news_summary", [])
        news_list = self.query_one("#news-list", ListView)
        news_list.clear()
        self.news_urls = []  # Reset URLs list

        if not news:
            news_list.append(ListItem(Static("No news articles available")))
            return

        for article in news:
            title = article.get("title", "No title")
            source = article.get("source", "Unknown")
            date = article.get("published_at", "")[:10] if article.get("published_at") else ""
            description = article.get("description", "")
            url = article.get("url", "")

            # Store the URL for this item
            self.news_urls.append(url)

            # Build content with title, metadata, and description
            content = f"[bold]{title}[/bold]\n[dim]{source} - {date}[/dim]"
            if description:
                # Truncate long descriptions and clean up whitespace
                desc_clean = " ".join(description.split())
                desc_truncated = desc_clean[:200] + "..." if len(desc_clean) > 200 else desc_clean
                content += f"\n{desc_truncated}"
            if url:
                content += "\n[dim italic]Press Enter to open in browser[/dim italic]"

            news_list.append(ListItem(Static(content)))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle news item selection - open URL in browser"""
        # Only handle selections from the news list
        if event.list_view.id != "news-list":
            return

        # Get the index of the selected item
        index = event.list_view.index
        if index is not None and 0 <= index < len(self.news_urls):
            url = self.news_urls[index]
            if url:
                webbrowser.open(url)
                self.notify(f"Opening article in browser...")

    # Actions
    async def action_refresh(self) -> None:
        """Refresh current stock"""
        if self.ticker:
            await self.load_stock(self.ticker)

    def action_add_stock(self) -> None:
        """Show add stock dialog"""
        def handle_result(result: str | None) -> None:
            if result:
                # Schedule the async load_stock call
                self.call_later(lambda: self.run_worker(
                    lambda: self._analyze_stock(result),
                    exclusive=True,
                    thread=True
                ))
                # Update header immediately
                self.ticker = result.upper()
                stock_info = self.query_one("#stock-info", Static)
                stock_info.update(f"Loading {self.ticker}...")

        self.push_screen(AddStockModal(), handle_result)

    def action_toggle_compare(self) -> None:
        """Toggle comparison mode"""
        self.compare_mode = not self.compare_mode
        self.notify(f"Comparison mode: {'ON' if self.compare_mode else 'OFF'}")

    def action_tab_1(self) -> None:
        self.query_one(TabbedContent).active = "tab-overview"

    def action_tab_2(self) -> None:
        self.query_one(TabbedContent).active = "tab-technical"

    def action_tab_3(self) -> None:
        self.query_one(TabbedContent).active = "tab-fundamental"

    def action_tab_4(self) -> None:
        self.query_one(TabbedContent).active = "tab-sentiment"

    def action_tab_5(self) -> None:
        self.query_one(TabbedContent).active = "tab-transcript"

    def action_tab_6(self) -> None:
        self.query_one(TabbedContent).active = "tab-news"
