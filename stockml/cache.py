"""Cache layer for StockML reports"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class ReportCache:
    """File-based cache for stock analysis reports

    Reports are cached per symbol and expire after a configurable duration.
    Cache is stored in ~/.stockml_cache/ by default.
    """

    DEFAULT_CACHE_DIR = Path.home() / ".stockml_cache"
    DEFAULT_MAX_AGE_DAYS = 7

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS
    ):
        """Initialize the cache

        Args:
            cache_dir: Directory to store cache files. Defaults to ~/.stockml_cache/
            max_age_days: Maximum age of cached data in days. Defaults to 7.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.max_age = timedelta(days=max_age_days)
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str, period: str = "1y") -> Path:
        """Get the cache file path for a symbol

        Args:
            symbol: Stock ticker symbol
            period: Analysis period (used to differentiate cache entries)

        Returns:
            Path to the cache file
        """
        # Create a unique cache key based on symbol and period
        cache_key = f"{symbol.upper()}_{period}"
        return self.cache_dir / f"{cache_key}.json"

    def get(self, symbol: str, period: str = "1y") -> Optional[dict]:
        """Get cached report if available and not expired

        Args:
            symbol: Stock ticker symbol
            period: Analysis period

        Returns:
            Cached report dict or None if not available/expired
        """
        cache_path = self._get_cache_path(symbol, period)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)

            # Check if cache has required metadata
            if "cached_at" not in cached_data or "report" not in cached_data:
                return None

            # Parse cached timestamp
            cached_at = datetime.fromisoformat(cached_data["cached_at"])

            # Check if cache is expired
            if datetime.now() - cached_at > self.max_age:
                # Cache expired, remove it
                self._remove_cache(cache_path)
                return None

            # Add cache metadata to report for UI display
            report = cached_data["report"]
            report["_cache_info"] = {
                "cached_at": cached_data["cached_at"],
                "cache_age_hours": round((datetime.now() - cached_at).total_seconds() / 3600, 1),
                "from_cache": True
            }

            return report

        except (json.JSONDecodeError, KeyError, ValueError):
            # Invalid cache file, remove it
            self._remove_cache(cache_path)
            return None

    def set(self, symbol: str, report: dict, period: str = "1y") -> None:
        """Store a report in the cache

        Args:
            symbol: Stock ticker symbol
            report: The report dict to cache
            period: Analysis period
        """
        cache_path = self._get_cache_path(symbol, period)

        # Remove any existing cache info from the report before storing
        report_to_cache = {k: v for k, v in report.items() if not k.startswith("_cache")}

        cached_data = {
            "symbol": symbol.upper(),
            "period": period,
            "cached_at": datetime.now().isoformat(),
            "report": report_to_cache
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(cached_data, f, indent=2, default=str)
        except (IOError, TypeError) as e:
            # Log error but don't fail - caching is optional
            import sys
            print(f"Warning: Failed to cache report: {e}", file=sys.stderr)

    def _remove_cache(self, cache_path: Path) -> None:
        """Remove a cache file

        Args:
            cache_path: Path to the cache file to remove
        """
        try:
            cache_path.unlink(missing_ok=True)
        except IOError:
            pass

    def invalidate(self, symbol: str, period: str = "1y") -> bool:
        """Invalidate (remove) cached data for a symbol

        Args:
            symbol: Stock ticker symbol
            period: Analysis period

        Returns:
            True if cache was removed, False if it didn't exist
        """
        cache_path = self._get_cache_path(symbol, period)
        if cache_path.exists():
            self._remove_cache(cache_path)
            return True
        return False

    def invalidate_all(self) -> int:
        """Remove all cached reports

        Returns:
            Number of cache files removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except IOError:
                pass
        return count

    def get_cache_info(self, symbol: str, period: str = "1y") -> Optional[dict]:
        """Get information about a cached entry without loading the full report

        Args:
            symbol: Stock ticker symbol
            period: Analysis period

        Returns:
            Dict with cache info or None if not cached
        """
        cache_path = self._get_cache_path(symbol, period)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)

            cached_at = datetime.fromisoformat(cached_data["cached_at"])
            age = datetime.now() - cached_at
            expires_in = self.max_age - age

            return {
                "symbol": cached_data.get("symbol"),
                "cached_at": cached_data["cached_at"],
                "age_hours": round(age.total_seconds() / 3600, 1),
                "age_days": round(age.total_seconds() / 86400, 2),
                "expires_in_hours": max(0, round(expires_in.total_seconds() / 3600, 1)),
                "expired": age > self.max_age,
                "file_size_kb": round(cache_path.stat().st_size / 1024, 1)
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def list_cached(self) -> list:
        """List all cached symbols with their cache info

        Returns:
            List of cache info dicts
        """
        cached = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)

                symbol = cached_data.get("symbol", cache_file.stem)
                period = cached_data.get("period", "1y")

                info = self.get_cache_info(symbol, period)
                if info:
                    cached.append(info)
            except (json.JSONDecodeError, KeyError):
                continue

        return cached

    def cleanup_expired(self) -> int:
        """Remove all expired cache entries

        Returns:
            Number of expired entries removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)

                cached_at = datetime.fromisoformat(cached_data["cached_at"])
                if datetime.now() - cached_at > self.max_age:
                    cache_file.unlink()
                    count += 1
            except (json.JSONDecodeError, KeyError, ValueError, IOError):
                # Invalid or unreadable cache file, remove it
                try:
                    cache_file.unlink()
                    count += 1
                except IOError:
                    pass

        return count
