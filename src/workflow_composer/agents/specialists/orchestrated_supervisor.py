"""
Orchestrator-Enhanced Supervisor
================================

Integrates NVIDIA Orchestrator-8B with the existing SupervisorAgent
for intelligent model/tool routing.

This provides a drop-in replacement that uses the trained orchestrator
to make cost-efficient routing decisions.

Usage:
    from workflow_composer.agents import OrchestratedSupervisor
    
    # Use with Orchestrator-8B
    supervisor = OrchestratedSupervisor(
        use_orchestrator=True,
        prefer_local=True,
        max_cost=0.50
    )
    
    result = await supervisor.execute("Generate RNA-seq workflow")
    print(f"Cost: ${result.metadata['cost']:.4f}")
    print(f"Models used: {result.metadata['models_used']}")
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from .supervisor import SupervisorAgent, WorkflowResult, WorkflowState
from .planner import WorkflowPlan
from workflow_composer.llm.orchestrator_8b import (
    Orchestrator8B,
    OrchestratorConfig,
    OrchestrationResult,
    ModelTier,
    BIOPIPELINE_TOOLS
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratedResult(WorkflowResult):
    """Extended result with orchestration metadata."""
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrchestratedSupervisor(SupervisorAgent):
    """
    SupervisorAgent enhanced with Orchestrator-8B routing.
    
    The orchestrator decides:
    - Which model tier to use for each phase
    - Whether to use local or cloud models
    - How to minimize cost while maintaining quality
    """
    
    def __init__(
        self,
        router=None,
        knowledge_base=None,
        use_orchestrator: bool = True,
        prefer_local: bool = True,
        max_cost: float = 1.0,
        orchestrator_backend: str = "vllm"
    ):
        """
        Initialize orchestrated supervisor.
        
        Args:
            router: LLM provider router (used when orchestrator disabled)
            knowledge_base: Knowledge base for RAG
            use_orchestrator: Whether to use Orchestrator-8B routing
            prefer_local: Prefer local models when possible
            max_cost: Maximum cost per query
            orchestrator_backend: Backend for orchestrator ("vllm", "transformers", "api")
        """
        super().__init__(router, knowledge_base)
        
        self.use_orchestrator = use_orchestrator
        self._orchestrator: Optional[Orchestrator8B] = None
        self._orchestrator_initialized = False
        
        # Store config for lazy initialization
        self._orch_config = OrchestratorConfig(
            inference_backend=orchestrator_backend,
            prefer_local=prefer_local,
            max_cost_per_query=max_cost,
            tool_catalog=BIOPIPELINE_TOOLS
        )
        
        # Tracking
        self._models_used: List[str] = []
        self._total_cost: float = 0.0
    
    async def _ensure_orchestrator(self):
        """Lazy initialization of orchestrator."""
        if not self.use_orchestrator:
            return
        
        if self._orchestrator_initialized:
            return
        
        try:
            self._orchestrator = Orchestrator8B(self._orch_config)
            await self._orchestrator.initialize()
            self._orchestrator_initialized = True
            logger.info("Orchestrator-8B initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Orchestrator-8B: {e}")
            logger.info("Falling back to heuristic routing")
            self._orchestrator = Orchestrator8B(self._orch_config)
            # Don't mark as initialized - will use heuristics
    
    async def execute(self, query: str, output_dir: str = None) -> OrchestratedResult:
        """
        Execute with orchestrator routing.
        
        The orchestrator decides which models to use at each phase.
        """
        await self._ensure_orchestrator()
        
        self._models_used = []
        self._total_cost = 0.0
        
        if self._orchestrator and self._orchestrator_initialized:
            # Full orchestrator mode
            result = await self._execute_with_orchestrator(query, output_dir)
        elif self._orchestrator:
            # Heuristic mode (orchestrator not fully initialized)
            result = await self._execute_with_heuristics(query, output_dir)
        else:
            # Fallback to base implementation
            base_result = await super().execute(query, output_dir)
            result = OrchestratedResult(
                success=base_result.success,
                plan=base_result.plan,
                code=base_result.code,
                config=base_result.config,
                documentation=base_result.documentation,
                validation_passed=base_result.validation_passed,
                validation_issues=base_result.validation_issues,
                output_files=base_result.output_files,
                metadata={"routing": "fallback", "cost": 0.0}
            )
        
        return result
    
    async def _execute_with_orchestrator(
        self,
        query: str,
        output_dir: str = None
    ) -> OrchestratedResult:
        """Execute with full Orchestrator-8B routing."""
        
        # Let orchestrator decide the execution plan
        orch_result = await self._orchestrator.route_and_execute(
            query=query,
            context=f"BioPipelines workflow generation. Output dir: {output_dir}"
        )
        
        self._total_cost += orch_result.cost
        
        if not orch_result.success:
            logger.warning("Orchestrator failed, falling back to standard execution")
            return await self._execute_with_heuristics(query, output_dir)
        
        # Parse orchestrator output and build result
        # The orchestrator should have coordinated the tool calls
        plan = self._extract_plan_from_result(orch_result)
        code = self._extract_code_from_result(orch_result)
        
        # If orchestrator didn't complete full workflow, fill in gaps
        if not plan:
            logger.info("Orchestrator didn't provide plan, generating...")
            plan = await self.planner.create_plan(query)
        
        if not code:
            logger.info("Orchestrator didn't provide code, generating...")
            code = await self.codegen.generate(plan)
        
        config = self.codegen.generate_config(plan) if plan else ""
        
        # Validate
        validation = await self.validator.validate(code)
        
        # Document
        readme = await self.docs.generate_readme(plan, code) if plan else ""
        
        result = OrchestratedResult(
            success=True,
            plan=plan,
            code=code,
            config=config,
            documentation=readme,
            validation_passed=validation.valid,
            validation_issues=validation.issues,
            output_files={},
            metadata={
                "routing": "orchestrator-8b",
                "cost": self._total_cost,
                "models_used": self._models_used,
                "orchestrator_reasoning": orch_result.orchestrator_reasoning,
                "turns": orch_result.turns,
                "tier_used": orch_result.tier_used.value
            }
        )
        
        if output_dir:
            result.output_files = self._write_outputs(output_dir, result)
        
        return result
    
    async def _execute_with_heuristics(
        self,
        query: str,
        output_dir: str = None
    ) -> OrchestratedResult:
        """Execute with heuristic-based routing."""
        
        # Get routing decision from orchestrator heuristics
        decision = self._orchestrator.get_routing_decision(query)
        
        logger.info(f"Heuristic routing: {decision.target_tier.value} -> {decision.target_model}")
        self._models_used.append(decision.target_model)
        
        # Adjust model selection based on tier
        if decision.target_tier == ModelTier.LOCAL_SMALL:
            # Use lightweight approach
            result = await self._execute_lightweight(query, output_dir)
        elif decision.target_tier == ModelTier.LOCAL_LARGE:
            # Use standard local execution
            result = await self._execute_standard(query, output_dir)
        else:
            # Use cloud-enhanced execution
            result = await self._execute_cloud_enhanced(query, output_dir)
        
        result.metadata["routing"] = "heuristic"
        result.metadata["estimated_cost"] = decision.estimated_cost
        result.metadata["model_tier"] = decision.target_tier.value
        
        return result
    
    async def _execute_lightweight(
        self,
        query: str,
        output_dir: str = None
    ) -> OrchestratedResult:
        """Lightweight execution for simple queries."""
        
        # Use sync methods which are template-based (no LLM cost)
        plan = self.planner.create_plan_sync(query)
        code = self.codegen.generate_sync(plan)
        config = self.codegen.generate_config(plan)
        validation = self.validator.validate_sync(code)
        readme = self.docs.generate_readme_sync(plan, code)
        
        result = OrchestratedResult(
            success=True,
            plan=plan,
            code=code,
            config=config,
            documentation=readme,
            validation_passed=validation.valid,
            validation_issues=validation.issues,
            output_files={},
            metadata={"cost": 0.0, "models_used": ["template-based"]}
        )
        
        if output_dir:
            result.output_files = self._write_outputs(output_dir, result)
        
        return result
    
    async def _execute_standard(
        self,
        query: str,
        output_dir: str = None
    ) -> OrchestratedResult:
        """Standard execution with local LLM."""
        base_result = await super().execute(query, output_dir)
        
        return OrchestratedResult(
            success=base_result.success,
            plan=base_result.plan,
            code=base_result.code,
            config=base_result.config,
            documentation=base_result.documentation,
            validation_passed=base_result.validation_passed,
            validation_issues=base_result.validation_issues,
            output_files=base_result.output_files,
            metadata={"cost": 0.01, "models_used": ["local-llm"]}
        )
    
    async def _execute_cloud_enhanced(
        self,
        query: str,
        output_dir: str = None
    ) -> OrchestratedResult:
        """Enhanced execution with cloud LLM for complex queries."""
        # Use async methods which leverage cloud LLM
        base_result = await super().execute(query, output_dir)
        
        return OrchestratedResult(
            success=base_result.success,
            plan=base_result.plan,
            code=base_result.code,
            config=base_result.config,
            documentation=base_result.documentation,
            validation_passed=base_result.validation_passed,
            validation_issues=base_result.validation_issues,
            output_files=base_result.output_files,
            metadata={"cost": 0.05, "models_used": ["cloud-llm"]}
        )
    
    def _extract_plan_from_result(self, result: OrchestrationResult) -> Optional[WorkflowPlan]:
        """Extract workflow plan from orchestrator result."""
        import json
        
        for tool_call in result.tool_calls:
            if tool_call.get("name") == "workflow_planner":
                # Tool result might be in response
                try:
                    # Parse from response
                    if "workflow_plan" in result.response.lower():
                        # Try to find JSON in response
                        import re
                        json_match = re.search(r'\{[^}]+\}', result.response)
                        if json_match:
                            data = json.loads(json_match.group())
                            return WorkflowPlan.from_dict(data)
                except:
                    pass
        return None
    
    def _extract_code_from_result(self, result: OrchestrationResult) -> Optional[str]:
        """Extract generated code from orchestrator result."""
        import re
        
        # Look for Nextflow code blocks
        code_match = re.search(
            r'```(?:nextflow|groovy)?\s*(.*?)```',
            result.response,
            re.DOTALL
        )
        
        if code_match:
            return code_match.group(1).strip()
        
        return None


# === Factory Function ===

def get_supervisor(
    use_orchestrator: bool = True,
    prefer_local: bool = True,
    max_cost: float = 1.0,
    **kwargs
) -> OrchestratedSupervisor:
    """
    Get a supervisor with optional Orchestrator-8B enhancement.
    
    Args:
        use_orchestrator: Whether to use Orchestrator-8B
        prefer_local: Prefer local models
        max_cost: Maximum cost per query
        **kwargs: Additional arguments for SupervisorAgent
        
    Returns:
        Configured OrchestratedSupervisor
    """
    return OrchestratedSupervisor(
        use_orchestrator=use_orchestrator,
        prefer_local=prefer_local,
        max_cost=max_cost,
        **kwargs
    )
