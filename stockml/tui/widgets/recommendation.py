"""Recommendation badge widget"""

from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Container


class RecommendationWidget(Static):
    """Display recommendation with colored badge"""

    DEFAULT_CSS = """
    RecommendationWidget {
        width: 100%;
        height: auto;
        padding: 1 2;
        border: solid $primary;
        text-align: center;
    }

    RecommendationWidget.strong-buy {
        background: $success 30%;
        border: solid $success;
    }

    RecommendationWidget.buy {
        background: $success 15%;
        border: solid $success;
    }

    RecommendationWidget.hold {
        background: $warning 20%;
        border: solid $warning;
    }

    RecommendationWidget.sell {
        background: $error 15%;
        border: solid $error;
    }

    RecommendationWidget.strong-sell {
        background: $error 30%;
        border: solid $error;
    }

    RecommendationWidget .action {
        text-style: bold;
    }

    RecommendationWidget .confidence {
        color: $text-muted;
    }
    """

    def __init__(self, action: str = "HOLD", confidence: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.action = action
        self.confidence = confidence

    def compose(self) -> ComposeResult:
        yield Static(self._render_content())

    def _render_content(self) -> str:
        bar_width = 20
        filled = int((self.confidence / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        return f"""
[bold]{self.action}[/bold]

{bar}

Confidence: {self.confidence}%
"""

    def on_mount(self) -> None:
        # Set CSS class based on action
        action_class = self.action.lower().replace("_", "-")
        self.add_class(action_class)

    def update_recommendation(self, action: str, confidence: int) -> None:
        """Update the recommendation display"""
        # Remove old class
        old_class = self.action.lower().replace("_", "-")
        self.remove_class(old_class)

        self.action = action
        self.confidence = confidence

        # Add new class
        new_class = action.lower().replace("_", "-")
        self.add_class(new_class)

        # Update content
        self.query_one(Static).update(self._render_content())
