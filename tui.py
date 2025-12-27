#!/usr/bin/env python3
"""StockML TUI - Terminal User Interface Entry Point"""

import argparse
import sys

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="StockML TUI - Interactive Stock Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tui.py AAPL              # Analyze Apple stock
  python tui.py                   # Start without a stock (add via 'a' key)

Keyboard shortcuts:
  q     Quit
  r     Refresh current stock
  a     Add/switch stock
  c     Toggle comparison mode
  1-6   Switch between tabs
  Tab   Navigate between elements
        """
    )

    parser.add_argument(
        "ticker",
        nargs="?",
        help="Stock ticker symbol to analyze (e.g., AAPL)"
    )

    args = parser.parse_args()

    try:
        from stockml.tui import StockMLApp
    except ImportError as e:
        print(f"Error: Could not import StockML TUI: {e}", file=sys.stderr)
        print("Make sure textual is installed: pip install textual", file=sys.stderr)
        sys.exit(1)

    app = StockMLApp(ticker=args.ticker)
    app.run()


if __name__ == "__main__":
    main()
