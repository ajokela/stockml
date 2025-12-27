"""Metrics panel widget for displaying key-value pairs"""

from textual.widgets import Static


class MetricsPanel(Static):
    """Display a panel of metrics with labels and values"""

    DEFAULT_CSS = """
    MetricsPanel {
        width: 100%;
        height: auto;
        padding: 1 2;
        border: solid $primary;
    }
    """

    def __init__(self, title: str = "", metrics: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.metrics = metrics or {}

    def on_mount(self) -> None:
        self._update_display()

    def _format_value(self, value, format_type: str = None) -> str:
        """Format value based on type"""
        if value is None:
            return "--"

        if format_type == "percent":
            return f"{value:.2f}%"
        elif format_type == "currency":
            return f"${value:.2f}"
        elif format_type == "currency_large":
            if value >= 1e12:
                return f"${value/1e12:.2f}T"
            elif value >= 1e9:
                return f"${value/1e9:.2f}B"
            elif value >= 1e6:
                return f"${value/1e6:.2f}M"
            else:
                return f"${value:,.0f}"
        elif format_type == "number":
            return f"{value:.2f}"
        elif format_type == "int":
            return f"{int(value):,}"
        else:
            return str(value)

    def _update_display(self) -> None:
        lines = []
        if self.title:
            lines.append(f"[bold]{self.title}[/bold]")
            lines.append("")

        for label, value_info in self.metrics.items():
            if isinstance(value_info, tuple):
                value, format_type = value_info
            else:
                value, format_type = value_info, None

            formatted = self._format_value(value, format_type)
            lines.append(f"{label}: {formatted}")

        if not self.metrics:
            lines.append("No data available")

        self.update("\n".join(lines))

    def update_metrics(self, metrics: dict, title: str = None) -> None:
        """Update the metrics"""
        if title is not None:
            self.title = title
        self.metrics = metrics
        self._update_display()
