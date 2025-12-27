"""Price targets widget"""

from textual.widgets import Static


class PriceTargetsWidget(Static):
    """Display price targets panel"""

    DEFAULT_CSS = """
    PriceTargetsWidget {
        width: 100%;
        height: auto;
        padding: 1 2;
        border: solid $primary;
    }
    """

    def __init__(
        self,
        current_price: float = 0,
        fair_value: float = None,
        target_buy: float = None,
        target_sell: float = None,
        stop_loss: float = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.current_price = current_price
        self.fair_value = fair_value
        self.target_buy = target_buy
        self.target_sell = target_sell
        self.stop_loss = stop_loss

    def on_mount(self) -> None:
        self._update_display()

    def _format_price(self, price: float, label: str, show_diff: bool = False) -> str:
        if price is None:
            return f"{label}: --"

        if show_diff and self.current_price:
            diff = ((price - self.current_price) / self.current_price) * 100
            diff_str = f"({diff:+.1f}%)"
            if diff > 0:
                return f"{label}: [green]${price:.2f}[/green] {diff_str}"
            elif diff < 0:
                return f"{label}: [red]${price:.2f}[/red] {diff_str}"
            else:
                return f"{label}: ${price:.2f} {diff_str}"
        else:
            return f"{label}: ${price:.2f}"

    def _update_display(self) -> None:
        lines = ["[bold]Price Targets[/bold]", ""]

        if self.fair_value:
            lines.append(self._format_price(self.fair_value, "Fair Value", show_diff=True))

        if self.target_buy:
            lines.append(self._format_price(self.target_buy, "Target Buy"))

        if self.target_sell:
            lines.append(self._format_price(self.target_sell, "Target Sell"))

        if self.stop_loss:
            lines.append(self._format_price(self.stop_loss, "Stop Loss"))

        if len(lines) == 2:
            lines.append("No price targets available")

        self.update("\n".join(lines))

    def update_targets(
        self,
        current_price: float = None,
        fair_value: float = None,
        target_buy: float = None,
        target_sell: float = None,
        stop_loss: float = None
    ) -> None:
        """Update price targets"""
        if current_price is not None:
            self.current_price = current_price
        if fair_value is not None:
            self.fair_value = fair_value
        if target_buy is not None:
            self.target_buy = target_buy
        if target_sell is not None:
            self.target_sell = target_sell
        if stop_loss is not None:
            self.stop_loss = stop_loss
        self._update_display()
