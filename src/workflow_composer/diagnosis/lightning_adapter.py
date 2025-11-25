"""
Lightning.ai LLM adapter for error diagnosis.

Integrates with Lightning.ai's free tier models for
pipeline error analysis.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .categories import ErrorCategory, FixRiskLevel, FixSuggestion, ErrorDiagnosis
from .prompts import DIAGNOSIS_PROMPT_SIMPLE, SYSTEM_PROMPT_DIAGNOSIS

logger = logging.getLogger(__name__)


@dataclass
class LightningConfig:
    """Configuration for Lightning.ai."""
    api_key: str
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    endpoint: str = "https://api.lightning.ai/v1/chat/completions"
    max_tokens: int = 1024
    temperature: float = 0.2


class LightningDiagnosisAdapter:
    """
    Lightning.ai adapter for error diagnosis.
    
    Uses Lightning.ai's inference API with free tier models
    like Llama, Mistral, etc.
    
    Environment Variables:
        LIGHTNING_API_KEY: Lightning.ai API key
        LIGHTNING_MODEL: Model to use (default: llama-3)
        
    Example:
        adapter = LightningDiagnosisAdapter()
        diagnosis = await adapter.diagnose(error_log, failed_process)
    """
    
    # Available free-tier models on Lightning.ai
    AVAILABLE_MODELS = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        "mistralai/Mistral-7B-Instruct-v0.2",
        "codellama/CodeLlama-7b-Instruct-hf",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize Lightning.ai adapter.
        
        Args:
            api_key: Lightning.ai API key (or from LIGHTNING_API_KEY env)
            model: Model name (or from LIGHTNING_MODEL env)
        """
        self.api_key = api_key or os.getenv("LIGHTNING_API_KEY")
        self.model = model or os.getenv(
            "LIGHTNING_MODEL", 
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        
        if not self.api_key:
            logger.warning(
                "LIGHTNING_API_KEY not set - Lightning.ai will be unavailable"
            )
    
    def is_available(self) -> bool:
        """Check if Lightning.ai is available."""
        return bool(self.api_key)
    
    async def diagnose(
        self,
        error_log: str,
        failed_process: str = "Unknown",
        workflow_name: str = "Unknown",
    ) -> Optional[ErrorDiagnosis]:
        """
        Diagnose an error using Lightning.ai.
        
        Args:
            error_log: Error log content
            failed_process: Name of failed process
            workflow_name: Name of the workflow
            
        Returns:
            ErrorDiagnosis or None if analysis fails
        """
        if not self.is_available():
            logger.warning("Lightning.ai not available")
            return None
        
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed - run: pip install httpx")
            return None
        
        prompt = DIAGNOSIS_PROMPT_SIMPLE.format(
            error_log=error_log[:4000],  # Limit for free tier
            failed_process=failed_process,
        )
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_DIAGNOSIS},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.lightning.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                return self._parse_response(content, error_log, failed_process)
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Lightning.ai API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Lightning.ai diagnosis failed: {e}")
            return None
    
    def diagnose_sync(
        self,
        error_log: str,
        failed_process: str = "Unknown",
        workflow_name: str = "Unknown",
    ) -> Optional[ErrorDiagnosis]:
        """
        Synchronous version of diagnose.
        
        Args:
            error_log: Error log content
            failed_process: Name of failed process
            workflow_name: Name of the workflow
            
        Returns:
            ErrorDiagnosis or None
        """
        if not self.is_available():
            return None
        
        try:
            import requests
        except ImportError:
            logger.error("requests not installed")
            return None
        
        prompt = DIAGNOSIS_PROMPT_SIMPLE.format(
            error_log=error_log[:4000],
            failed_process=failed_process,
        )
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_DIAGNOSIS},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        }
        
        try:
            response = requests.post(
                "https://api.lightning.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self._parse_response(content, error_log, failed_process)
            
        except Exception as e:
            logger.error(f"Lightning.ai diagnosis failed: {e}")
            return None
    
    def _parse_response(
        self,
        content: str,
        error_log: str,
        failed_process: str,
    ) -> Optional[ErrorDiagnosis]:
        """Parse LLM response into ErrorDiagnosis."""
        try:
            # Extract JSON from response
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            # Map category
            category_str = data.get("error_category", "unknown").lower()
            category_map = {
                "file_not_found": ErrorCategory.FILE_NOT_FOUND,
                "out_of_memory": ErrorCategory.OUT_OF_MEMORY,
                "container_error": ErrorCategory.CONTAINER_ERROR,
                "permission_denied": ErrorCategory.PERMISSION_DENIED,
                "dependency_missing": ErrorCategory.DEPENDENCY_MISSING,
                "tool_error": ErrorCategory.TOOL_ERROR,
                "network_error": ErrorCategory.NETWORK_ERROR,
                "slurm_error": ErrorCategory.SLURM_ERROR,
                "data_format_error": ErrorCategory.DATA_FORMAT_ERROR,
            }
            category = category_map.get(category_str, ErrorCategory.UNKNOWN)
            
            # Parse fixes
            fixes = []
            for fix_data in data.get("suggested_fixes", []):
                risk_str = fix_data.get("risk_level", "medium").lower()
                risk_map = {
                    "safe": FixRiskLevel.SAFE,
                    "low": FixRiskLevel.LOW,
                    "medium": FixRiskLevel.MEDIUM,
                    "high": FixRiskLevel.HIGH,
                }
                risk = risk_map.get(risk_str, FixRiskLevel.MEDIUM)
                
                fixes.append(FixSuggestion(
                    description=fix_data.get("description", ""),
                    command=fix_data.get("command"),
                    auto_executable=fix_data.get("auto_executable", False),
                    risk_level=risk,
                ))
            
            return ErrorDiagnosis(
                category=category,
                root_cause=data.get("root_cause", "Unknown"),
                user_explanation=data.get("user_explanation", data.get("root_cause", "")),
                confidence=float(data.get("confidence", 0.7)),
                log_excerpt=data.get("log_excerpt", error_log[:500]),
                suggested_fixes=fixes,
                failed_process=failed_process,
                llm_provider_used=f"lightning.ai/{self.model}",
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Lightning.ai response: {e}")
            logger.debug(f"Response content: {content[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None


def get_lightning_adapter() -> Optional[LightningDiagnosisAdapter]:
    """Get a Lightning.ai adapter if available."""
    adapter = LightningDiagnosisAdapter()
    if adapter.is_available():
        return adapter
    return None
