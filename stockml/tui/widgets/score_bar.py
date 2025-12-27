"""Score bar widget for displaying analysis scores"""

from textual.widgets import Static


class ScoreBar(Static):
    """Horizontal bar showing score from -100 to +100"""

    DEFAULT_CSS = """
    ScoreBar {
        width: 100%;
        height: 1;
        margin: 0 1;
    }

    ScoreBar.positive {
        color: $success;
    }

    ScoreBar.negative {
        color: $error;
    }

    ScoreBar.neutral {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        label: str,
        score: int = 0,
        description: str = "",
        bar_width: int = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.label = label
        self.score = score
        self.description = description
        self.bar_width = bar_width

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        # Calculate bar position (score from -100 to +100, bar shows 0 in middle)
        # Normalize to 0-100 scale for display
        normalized = (self.score + 100) / 2  # 0 = -100, 50 = 0, 100 = +100

        half_width = self.bar_width // 2
        center = half_width

        if self.score >= 0:
            # Positive: fill from center to right
            filled_right = int((self.score / 100) * half_width)
            bar = "░" * half_width + "█" * filled_right + "░" * (half_width - filled_right)
            self.add_class("positive")
            self.remove_class("negative")
            self.remove_class("neutral")
        elif self.score < 0:
            # Negative: fill from center to left
            filled_left = int((abs(self.score) / 100) * half_width)
            bar = "░" * (half_width - filled_left) + "█" * filled_left + "░" * half_width
            self.add_class("negative")
            self.remove_class("positive")
            self.remove_class("neutral")
        else:
            bar = "░" * self.bar_width
            self.add_class("neutral")
            self.remove_class("positive")
            self.remove_class("negative")

        # Format: "Label:    ████████░░░░░░░░░░░░  +45  description"
        score_str = f"{self.score:+d}" if self.score != 0 else "0"
        label_padded = f"{self.label}:".ljust(14)

        content = f"{label_padded}{bar}  {score_str:>4}  {self.description}"
        self.update(content)

    def update_score(self, score: int, description: str = "") -> None:
        """Update the score and description"""
        self.score = score
        self.description = description
        self._update_display()
