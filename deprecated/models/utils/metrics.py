"""
Usage metrics tracking for model providers.

Tracks token usage, latency, and costs across providers.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict


@dataclass
class ProviderUsage:
    """Usage statistics for a single provider."""
    provider_id: str
    requests: int = 0
    tokens_used: int = 0
    total_latency_ms: float = 0
    errors: int = 0
    last_used: Optional[str] = None
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.requests == 0:
            return 0
        return self.total_latency_ms / self.requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class UsageMetrics:
    """
    Tracks usage metrics across all providers.
    
    Persists data to JSON for historical analysis.
    """
    
    metrics_dir: Path = field(default_factory=lambda: Path("logs/model_metrics"))
    
    def __post_init__(self):
        self.metrics_dir = Path(self.metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self._daily: Dict[str, ProviderUsage] = {}
        self._session: Dict[str, ProviderUsage] = {}
        self._load_today()
    
    def _get_today_file(self) -> Path:
        """Get path to today's metrics file."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.metrics_dir / f"usage_{today}.json"
    
    def _load_today(self):
        """Load today's metrics if they exist."""
        today_file = self._get_today_file()
        if today_file.exists():
            try:
                with open(today_file) as f:
                    data = json.load(f)
                    for provider_id, usage in data.get("providers", {}).items():
                        self._daily[provider_id] = ProviderUsage(
                            provider_id=provider_id,
                            **{k: v for k, v in usage.items() 
                               if k != "avg_latency_ms" and k != "provider_id"}
                        )
            except (json.JSONDecodeError, Exception):
                pass
    
    def _save_today(self):
        """Save today's metrics."""
        today_file = self._get_today_file()
        data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "last_updated": datetime.now().isoformat(),
            "providers": {
                provider_id: usage.to_dict()
                for provider_id, usage in self._daily.items()
            },
        }
        with open(today_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def record_request(
        self,
        provider_id: str,
        tokens_used: int = 0,
        latency_ms: float = 0,
        success: bool = True,
    ):
        """
        Record a request to a provider.
        
        Args:
            provider_id: Provider that handled the request
            tokens_used: Number of tokens consumed
            latency_ms: Request latency in milliseconds
            success: Whether the request succeeded
        """
        now = datetime.now().isoformat()
        
        # Update daily stats
        if provider_id not in self._daily:
            self._daily[provider_id] = ProviderUsage(provider_id=provider_id)
        
        usage = self._daily[provider_id]
        usage.requests += 1
        usage.tokens_used += tokens_used
        usage.total_latency_ms += latency_ms
        usage.last_used = now
        
        if not success:
            usage.errors += 1
        
        # Update session stats
        if provider_id not in self._session:
            self._session[provider_id] = ProviderUsage(provider_id=provider_id)
        
        session = self._session[provider_id]
        session.requests += 1
        session.tokens_used += tokens_used
        session.total_latency_ms += latency_ms
        session.last_used = now
        
        if not success:
            session.errors += 1
        
        # Persist
        self._save_today()
    
    def get_daily_usage(self) -> Dict[str, ProviderUsage]:
        """Get today's usage by provider."""
        return dict(self._daily)
    
    def get_session_usage(self) -> Dict[str, ProviderUsage]:
        """Get this session's usage by provider."""
        return dict(self._session)
    
    def get_usage_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get usage summary for the last N days.
        
        Args:
            days: Number of days to summarize
            
        Returns:
            Summary with totals per provider
        """
        totals: Dict[str, ProviderUsage] = {}
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            file_path = self.metrics_dir / f"usage_{date}.json"
            
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                        for provider_id, usage in data.get("providers", {}).items():
                            if provider_id not in totals:
                                totals[provider_id] = ProviderUsage(
                                    provider_id=provider_id
                                )
                            
                            totals[provider_id].requests += usage.get("requests", 0)
                            totals[provider_id].tokens_used += usage.get("tokens_used", 0)
                            totals[provider_id].total_latency_ms += usage.get("total_latency_ms", 0)
                            totals[provider_id].errors += usage.get("errors", 0)
                except (json.JSONDecodeError, Exception):
                    pass
        
        return {
            "period_days": days,
            "providers": {
                provider_id: usage.to_dict()
                for provider_id, usage in totals.items()
            },
            "total_requests": sum(u.requests for u in totals.values()),
            "total_tokens": sum(u.tokens_used for u in totals.values()),
            "total_errors": sum(u.errors for u in totals.values()),
        }
    
    def estimate_costs(self, days: int = 30) -> Dict[str, float]:
        """
        Estimate costs based on token usage.
        
        Note: These are rough estimates. Actual costs depend on
        the specific models used.
        
        Args:
            days: Number of days to calculate
            
        Returns:
            Estimated costs per provider in USD
        """
        # Approximate costs per 1M tokens
        COST_PER_MILLION = {
            "lightning": 0.0,  # Free tier
            "gemini": 0.0,     # Free tier
            "openai": 2.50,    # GPT-4o-mini average
            "vllm": 0.0,       # Local (electricity cost not included)
            "github_copilot": 0.0,  # Subscription-based
        }
        
        summary = self.get_usage_summary(days)
        costs = {}
        
        for provider_id, usage in summary.get("providers", {}).items():
            tokens = usage.get("tokens_used", 0)
            cost_per_m = COST_PER_MILLION.get(provider_id, 1.0)
            costs[provider_id] = (tokens / 1_000_000) * cost_per_m
        
        return costs
    
    def print_usage_report(self, days: int = 7):
        """Print a formatted usage report."""
        summary = self.get_usage_summary(days)
        costs = self.estimate_costs(days)
        
        print("\n" + "=" * 60)
        print(f"MODEL USAGE REPORT ({days} days)")
        print("=" * 60 + "\n")
        
        for provider_id, usage in summary.get("providers", {}).items():
            print(f"📊 {provider_id.upper()}")
            print(f"   Requests: {usage.get('requests', 0):,}")
            print(f"   Tokens:   {usage.get('tokens_used', 0):,}")
            print(f"   Errors:   {usage.get('errors', 0)}")
            print(f"   Avg Latency: {usage.get('avg_latency_ms', 0):.0f}ms")
            print(f"   Est. Cost: ${costs.get(provider_id, 0):.4f}")
            print()
        
        print("-" * 60)
        print(f"TOTALS")
        print(f"   Total Requests: {summary.get('total_requests', 0):,}")
        print(f"   Total Tokens:   {summary.get('total_tokens', 0):,}")
        print(f"   Total Errors:   {summary.get('total_errors', 0)}")
        print(f"   Total Est. Cost: ${sum(costs.values()):.4f}")
        print("=" * 60 + "\n")


# Global metrics instance
_metrics: Optional[UsageMetrics] = None


def get_usage_tracker() -> UsageMetrics:
    """Get the global usage metrics tracker."""
    global _metrics
    if _metrics is None:
        _metrics = UsageMetrics()
    return _metrics
