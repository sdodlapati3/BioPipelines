"""
GitHub Copilot Provider Implementation.

GitHub Copilot provides AI-powered code assistance through the GitHub API.
This is primarily used for code suggestions and completions.
"""

import os
import time
from typing import Optional, Dict, Any

from .base import BaseProvider
from ..registry import ProviderConfig


class GitHubCopilotProvider(BaseProvider):
    """
    Provider for GitHub Copilot.
    
    Note: GitHub Copilot has a different API model than typical LLMs.
    It's primarily designed for IDE integration. This provider attempts
    to use the GitHub API for code-related tasks where possible.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.token = os.environ.get(config.env_key or "GITHUB_TOKEN")
    
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        GitHub Copilot completion.
        
        Note: Direct Copilot API access is limited. This provider
        primarily serves as a fallback indicator that Copilot
        integration is available through the IDE.
        """
        # GitHub Copilot doesn't have a direct completion API
        # Instead, we can use GitHub's code search and context features
        
        raise NotImplementedError(
            "GitHub Copilot direct API access is not available. "
            "Use Copilot through your IDE integration instead. "
            "For programmatic access, consider using the GitHub API "
            "for code search and repository operations."
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if GitHub token is available."""
        if not self.token:
            return {
                "available": False,
                "error": "GITHUB_TOKEN not configured",
            }
        
        try:
            import aiohttp
            
            start = time.time()
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.github.com/user",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    latency = (time.time() - start) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "available": True,
                            "latency_ms": latency,
                            "user": data.get("login"),
                            "note": "Use Copilot through IDE for completions",
                        }
                    else:
                        return {
                            "available": False,
                            "error": f"HTTP {response.status}",
                            "latency_ms": latency,
                        }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
            }
