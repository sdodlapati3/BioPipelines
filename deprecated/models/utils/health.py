"""
Health checking utilities for model providers.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class HealthStatus:
    """Health status for a provider."""
    provider_id: str
    available: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    models_loaded: Optional[List[str]] = None
    checked_at: datetime = None
    
    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider_id": self.provider_id,
            "available": self.available,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "models_loaded": self.models_loaded,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
        }


async def check_provider(provider_id: str) -> HealthStatus:
    """
    Check health of a specific provider.
    
    Args:
        provider_id: Provider ID to check
        
    Returns:
        HealthStatus with check results
    """
    from ..registry import get_registry
    from ..router import ModelRouter
    
    registry = get_registry()
    provider = registry.get_provider(provider_id)
    
    if not provider:
        return HealthStatus(
            provider_id=provider_id,
            available=False,
            error=f"Unknown provider: {provider_id}",
        )
    
    if not provider.is_available():
        return HealthStatus(
            provider_id=provider_id,
            available=False,
            error="Provider not configured (missing API key or disabled)",
        )
    
    try:
        router = ModelRouter()
        client = router._get_provider_client(provider)
        result = await client.health_check()
        
        return HealthStatus(
            provider_id=provider_id,
            available=result.get("available", False),
            latency_ms=result.get("latency_ms"),
            error=result.get("error"),
            models_loaded=result.get("models_loaded"),
        )
    except Exception as e:
        return HealthStatus(
            provider_id=provider_id,
            available=False,
            error=str(e),
        )


async def check_all_providers() -> Dict[str, HealthStatus]:
    """
    Check health of all configured providers.
    
    Returns:
        Dictionary mapping provider IDs to HealthStatus
    """
    from ..registry import get_registry
    
    registry = get_registry()
    providers = registry.list_providers(available_only=False)
    
    # Check all providers in parallel
    tasks = [check_provider(p.id) for p in providers]
    results = await asyncio.gather(*tasks)
    
    return {status.provider_id: status for status in results}


def check_provider_sync(provider_id: str) -> HealthStatus:
    """Synchronous wrapper for check_provider."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(check_provider(provider_id))


def check_all_providers_sync() -> Dict[str, HealthStatus]:
    """Synchronous wrapper for check_all_providers."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(check_all_providers())


def print_health_report():
    """Print a formatted health report for all providers."""
    statuses = check_all_providers_sync()
    
    print("\n" + "=" * 60)
    print("MODEL PROVIDER HEALTH CHECK")
    print("=" * 60 + "\n")
    
    for provider_id, status in sorted(
        statuses.items(),
        key=lambda x: (not x[1].available, x[0])
    ):
        emoji = "✅" if status.available else "❌"
        latency = f" ({status.latency_ms:.0f}ms)" if status.latency_ms else ""
        
        print(f"{emoji} {provider_id:20}{latency}")
        
        if status.error:
            print(f"   Error: {status.error}")
        
        if status.models_loaded:
            print(f"   Models: {', '.join(status.models_loaded[:3])}")
            if len(status.models_loaded) > 3:
                print(f"           ... and {len(status.models_loaded) - 3} more")
        
        print()
    
    print("=" * 60)
    available = sum(1 for s in statuses.values() if s.available)
    print(f"Total: {available}/{len(statuses)} providers available")
    print("=" * 60 + "\n")
