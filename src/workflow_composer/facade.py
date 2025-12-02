"""
BioPipelines Façade
===================

The single, unified entry point for BioPipelines.

This provides a clean, simple API that hides the complexity of the underlying
components while providing access to all functionality.

Usage:
    from workflow_composer import BioPipelines
    
    # Initialize
    bp = BioPipelines()
    
    # Chat mode (natural language)
    response = bp.chat("Create an RNA-seq differential expression workflow")
    
    # Direct workflow generation
    workflow = bp.generate("RNA-seq differential expression", 
                           samples=["sample1.fq", "sample2.fq"])
    
    # Submit to cluster
    job = bp.submit(workflow)
    
    # Check status
    status = bp.status(job.id)
    
    # Scan for data
    samples = bp.scan_data("/path/to/fastq")
    
    # Search tools
    tools = bp.find_tools("alignment")
    
Advanced Usage:
    # Custom configuration
    bp = BioPipelines(
        llm_provider="openai",
        llm_model="gpt-4o",
        cluster="slurm",
    )
    
    # Access underlying components
    bp.composer.generate(...)
    bp.agent.process(...)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Iterator, TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)

# Type checking imports
if TYPE_CHECKING:
    from .core import Workflow, Tool, ParsedIntent
    from .agents import AgentResponse
    from .data import DataManifest


@dataclass
class JobStatus:
    """Status of a submitted job."""
    
    job_id: str
    """SLURM job ID or local process ID."""
    
    state: str
    """Job state: pending, running, completed, failed, cancelled."""
    
    progress: float
    """Progress 0.0 to 1.0."""
    
    message: str
    """Human-readable status message."""
    
    outputs: List[str] = field(default_factory=list)
    """List of output file paths."""
    
    errors: List[str] = field(default_factory=list)
    """List of error messages if any."""
    
    @property
    def is_complete(self) -> bool:
        """Check if job is finished (success or failure)."""
        return self.state in ("completed", "failed", "cancelled")
    
    @property
    def is_success(self) -> bool:
        """Check if job completed successfully."""
        return self.state == "completed"


@dataclass
class ChatResponse:
    """Response from chat interaction."""
    
    message: str
    """The assistant's response message."""
    
    workflow: Optional["Workflow"] = None
    """Generated workflow if any."""
    
    tools_used: List[str] = field(default_factory=list)
    """Tools that were executed."""
    
    suggestions: List[str] = field(default_factory=list)
    """Follow-up suggestions."""
    
    correlation_id: str = ""
    """Request correlation ID for tracking."""
    
    session_id: Optional[str] = None
    """Session ID for multi-turn conversations."""


class BioPipelines:
    """
    The unified entry point for BioPipelines.
    
    Provides a simple, consistent API for:
    - Natural language workflow generation
    - Data discovery and management
    - Job submission and monitoring
    - Tool catalog access
    - Smart LLM orchestration (local/cloud routing)
    
    Example:
        bp = BioPipelines()
        
        # Generate a workflow
        workflow = bp.generate("RNA-seq analysis for human samples")
        
        # Or use chat for interactive workflow creation
        response = bp.chat("I need to analyze ChIP-seq data")
        
        # Submit to cluster
        job = bp.submit(workflow, cluster="slurm")
        
        # Monitor
        status = bp.status(job.id)
        
        # Use orchestrator for smart LLM routing
        response = await bp.orchestrator.complete("Generate workflow")
    """
    
    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        cluster: str = "local",
        config_path: Optional[str] = None,
        orchestrator_preset: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize BioPipelines.
        
        Args:
            llm_provider: LLM provider ("lightning", "openai", "ollama", etc.)
            llm_model: Model name (provider-specific)
            cluster: Execution target ("local", "slurm")
            config_path: Path to configuration file
            orchestrator_preset: Preset for LLM orchestrator ("development", "production", "critical")
            **kwargs: Additional configuration options
        """
        # Lazy imports to avoid circular dependencies
        from .config import Config
        
        # Load configuration
        self._config = Config.load(config_path)
        
        # Override with explicit parameters
        if llm_provider:
            self._config.llm.default_provider = llm_provider
        if llm_model:
            self._config.llm.intent_parser_model = llm_model
        
        self._cluster = cluster
        self._orchestrator_preset = orchestrator_preset
        self._kwargs = kwargs
        
        # Lazy initialization
        self._composer = None
        self._agent = None
        self._orchestrator = None
        self._cost_tracker = None
        self._session_manager = None
        self._supervisor = None  # Multi-agent supervisor
        self._initialized = False
        
        logger.info(f"BioPipelines initialized (llm={llm_provider or 'default'})")
    
    def _ensure_initialized(self) -> None:
        """Lazily initialize components."""
        if self._initialized:
            return
        
        from .composer import Composer
        from .llm import get_llm
        
        # Initialize LLM
        provider = self._config.llm.default_provider
        model = self._config.llm.intent_parser_model
        llm = get_llm(provider, model)
        
        # Initialize composer
        self._composer = Composer(llm=llm, config=self._config)
        
        self._initialized = True
    
    @property
    def composer(self) -> "Composer":
        """Access the underlying Composer for advanced usage."""
        self._ensure_initialized()
        return self._composer
    
    @property
    def agent(self) -> "UnifiedAgent":
        """Access the underlying UnifiedAgent for advanced usage."""
        if self._agent is None:
            from .agents import UnifiedAgent
            self._agent = UnifiedAgent()
        return self._agent
    
    @property
    def chat_agent(self) -> "ChatAgent":
        """
        Access the professional ChatAgent for conversational AI.
        
        The ChatAgent provides:
        - Dialog state management
        - Scope detection (out-of-scope deflection)
        - A/B testing for response optimization
        - Analytics and metrics
        - Human handoff when needed
        - Rich response formatting
        
        This wraps UnifiedAgent and adds professional chat features.
        
        Example:
            response = bp.chat_agent.process_message(
                "Create an RNA-seq workflow",
                user_id="user123"
            )
        """
        if not hasattr(self, '_chat_agent') or self._chat_agent is None:
            from .agents.intent.chat_agent import ChatAgent, AgentConfig
            self._chat_agent = ChatAgent()
        return self._chat_agent
    
    @property
    def orchestrator(self) -> "ModelOrchestrator":
        """
        Access the LLM orchestrator for smart model routing.
        
        The orchestrator provides:
        - Automatic local/cloud model selection
        - Cost-aware routing
        - Fallback handling
        - Ensemble for critical tasks
        
        Example:
            # Get smart model routing
            response = await bp.orchestrator.complete("Generate workflow")
            print(f"Used: {response.provider}, Cost: ${response.cost:.4f}")
            
            # Check usage stats
            print(bp.orchestrator.stats)
        """
        if self._orchestrator is None:
            from .llm import get_orchestrator
            self._orchestrator = get_orchestrator(preset=self._orchestrator_preset)
        return self._orchestrator
    
    @property
    def cost_tracker(self) -> "CostTracker":
        """
        Access the cost tracker for budget management.
        
        Example:
            print(f"Total spent: ${bp.cost_tracker.total_cost:.2f}")
            print(bp.cost_tracker.summary())
        """
        if self._cost_tracker is None:
            from .llm import CostTracker
            self._cost_tracker = CostTracker()
        return self._cost_tracker
    
    @property
    def session_manager(self) -> "SessionManager":
        """
        Access the session manager for multi-turn conversations.
        
        Sessions provide:
        - Conversation history persistence
        - User preference learning
        - Context injection (organism, analysis type)
        
        Example:
            session = bp.session_manager.create_session("user_123")
            response = bp.chat("RNA-seq analysis", session_id=session.session_id)
        """
        if self._session_manager is None:
            from .agents.memory import SessionManager
            self._session_manager = SessionManager()
        return self._session_manager
    
    @property
    def supervisor(self) -> "SupervisorAgent":
        """
        Access the multi-agent supervisor for advanced workflow generation.
        
        The supervisor coordinates specialist agents:
        - PlannerAgent: Designs workflow architecture
        - CodeGenAgent: Generates Nextflow DSL2 code
        - ValidatorAgent: Reviews and validates code
        - DocAgent: Generates documentation
        - QCAgent: Quality control validation
        
        Example:
            # Full async workflow generation with multi-agent coordination
            result = await bp.supervisor.execute("RNA-seq differential expression")
            print(result.code)  # Generated main.nf
            print(result.documentation)  # Generated README.md
            
            # Streaming progress updates
            async for update in bp.supervisor.execute_streaming("ChIP-seq analysis"):
                print(f"{update['phase']}: {update['status']}")
                
            # Synchronous (template-based) generation
            result = bp.supervisor.execute_sync("variant calling pipeline")
        """
        if self._supervisor is None:
            from .agents.specialists import SupervisorAgent
            # Try to get router for LLM operations
            try:
                router = self.orchestrator
            except Exception:
                router = None
            
            # Try to get knowledge base for RAG
            try:
                from .agents.rag import KnowledgeBase
                knowledge_base = KnowledgeBase()
            except Exception:
                knowledge_base = None
            
            self._supervisor = SupervisorAgent(router=router, knowledge_base=knowledge_base)
        return self._supervisor
    
    async def generate_with_agents(
        self,
        description: str,
        output_dir: Optional[str] = None,
    ) -> "SpecialistWorkflowResult":
        """
        Generate a workflow using multi-agent coordination.
        
        This is the advanced workflow generation method that uses:
        - PlannerAgent to design the workflow architecture
        - CodeGenAgent to generate Nextflow DSL2 code
        - ValidatorAgent to review with fix loops
        - DocAgent to create documentation
        
        Args:
            description: Natural language description of the workflow
            output_dir: Directory to save generated files
            
        Returns:
            WorkflowResult with code, config, and documentation
            
        Example:
            result = await bp.generate_with_agents(
                "RNA-seq differential expression for human",
                output_dir="workflows/rnaseq"
            )
            if result.success:
                print(f"Generated {len(result.code.split(chr(10)))} lines of Nextflow code")
        """
        return await self.supervisor.execute(description, output_dir)
    
    def generate_with_agents_sync(
        self,
        description: str,
        output_dir: Optional[str] = None,
    ) -> "SpecialistWorkflowResult":
        """
        Synchronous version of generate_with_agents using templates.
        
        Example:
            result = bp.generate_with_agents_sync("ChIP-seq peak calling")
            print(result.code)
        """
        return self.supervisor.execute_sync(description, output_dir)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check system health and component availability.
        
        Returns:
            Dict with health status of each component
            
        Example:
            health = bp.health_check()
            if health["llm_available"]:
                print(f"Using LLM: {health['llm_provider']}")
        """
        result = {
            "llm_available": False,
            "llm_provider": None,
            "tools_available": False,
            "tool_count": 0,
            "orchestrator_available": False,
        }
        
        # Check LLM/orchestrator
        try:
            orch = self.orchestrator
            if orch:
                result["orchestrator_available"] = True
                # Try to get provider info
                if hasattr(orch, '_local_provider') and orch._local_provider:
                    result["llm_available"] = True
                    result["llm_provider"] = "local"
                elif hasattr(orch, '_cloud_provider') and orch._cloud_provider:
                    result["llm_available"] = True
                    result["llm_provider"] = "cloud"
        except Exception:
            pass
        
        # Check agent/tools
        try:
            agent = self.agent
            if agent and hasattr(agent, '_tools') and agent._tools:
                result["tools_available"] = True
                result["tool_count"] = len(agent._tools.tools) if hasattr(agent._tools, 'tools') else 0
        except Exception:
            pass
        
        return result
    
    # =========================================================================
    # Core API - Workflow Generation
    # =========================================================================
    
    def generate(
        self,
        description: str,
        samples: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        auto_create_modules: bool = True,
        **options
    ) -> "Workflow":
        """
        Generate a workflow from a natural language description.
        
        This is the primary method for creating bioinformatics workflows.
        
        Args:
            description: Natural language description of the analysis
            samples: Optional list of sample file paths
            output_dir: Directory to save the workflow
            auto_create_modules: Auto-create missing modules with LLM
            **options: Additional options passed to workflow generator
            
        Returns:
            Generated Workflow object
            
        Example:
            workflow = bp.generate(
                "RNA-seq differential expression for mouse",
                samples=["sample1.fq.gz", "sample2.fq.gz"],
                output_dir="workflows/rnaseq"
            )
        """
        self._ensure_initialized()
        
        # Create data manifest if samples provided
        data_manifest = None
        if samples:
            from .data import DataManifest, Sample
            manifest_samples = [
                Sample(path=Path(s), sample_id=Path(s).stem)
                for s in samples
            ]
            data_manifest = DataManifest(samples=manifest_samples)
        
        return self._composer.generate(
            description=description,
            output_dir=output_dir,
            auto_create_modules=auto_create_modules,
            data_manifest=data_manifest,
            **options
        )
    
    # =========================================================================
    # Chat Helpers (shared between chat and chat_stream)
    # =========================================================================
    
    def _prepare_chat_session(
        self,
        session_id: Optional[str],
        user_id: str,
        message: str,
    ):
        """
        Prepare session for chat interaction.
        
        Handles session creation/retrieval and message logging.
        
        Args:
            session_id: Optional session ID
            user_id: User identifier
            message: User's message
            
        Returns:
            Session object or None
        """
        session = None
        if session_id:
            session = self.session_manager.get_session(session_id)
            if not session:
                # Create new session if requested ID doesn't exist
                session = self.session_manager.create_session(user_id)
        
        # Add user message to session
        if session:
            self.session_manager.add_user_message(
                session.session_id,
                message,
                parsed_intent=None,
            )
        
        return session
    
    def _process_agent_query(self, message: str) -> "AgentResponse":
        """
        Process a message through the UnifiedAgent.
        
        This enables tool execution (scan data, search databases, etc.)
        
        Args:
            message: User's query
            
        Returns:
            AgentResponse with results
        """
        agent = self.agent
        return agent.process_sync(message)
    
    def _finalize_chat_session(
        self,
        session,
        response_message: str,
        workflow=None,
    ) -> None:
        """
        Finalize session after chat interaction.
        
        Adds the assistant's response to session history.
        
        Args:
            session: Session object or None
            response_message: Assistant's response
            workflow: Optional generated workflow
        """
        if session:
            self.session_manager.add_assistant_message(
                session.session_id,
                response_message,
                workflow=workflow,
            )
    
    def _was_tool_executed(self, result: "AgentResponse") -> bool:
        """
        Check if the agent executed a tool for the query.
        
        Args:
            result: AgentResponse from agent processing
            
        Returns:
            True if a tool was executed
        """
        return (
            (hasattr(result, 'tools_used') and result.tools_used) or
            (hasattr(result, 'data') and result.data) or
            (hasattr(result, 'response_type') and 
             str(result.response_type) not in ('ResponseType.INFO', 'ResponseType.ERROR'))
        )
    
    def _get_chat_system_prompt(self, session=None) -> str:
        """
        Get the system prompt for chat interactions.
        
        Args:
            session: Optional session for context
            
        Returns:
            System prompt string
        """
        system_prompt = """You are BioPipelines, an AI assistant for bioinformatics workflow generation.
You help users create, manage, and run bioinformatics pipelines.

IMPORTANT: You CAN access local files and run commands. When users ask about local data:
- Use 'scan data in /path' to scan directories
- Use 'search for <query>' to search databases
- Use 'show jobs' to list running jobs

Be helpful, concise, and technically accurate."""
        
        # Add session context if available
        if session:
            context = self.session_manager.get_session_context(session.session_id)
            if context.get("context_summary"):
                system_prompt += f"\n\nUser context: {context['context_summary']}"
        
        return system_prompt
    
    # =========================================================================
    # Core API - Chat Interface
    # =========================================================================
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Chat interface for natural language interaction.
        
        Uses the professional ChatAgent which provides:
        - Scope detection (out-of-scope deflection)
        - Dialog state management
        - A/B testing for response optimization
        - Analytics and metrics
        - Human handoff when needed
        - Tool execution via UnifiedAgent
        
        Session Support:
        - If session_id is provided, conversation is tracked
        - User preferences are learned from interactions
        - Context (organism, analysis type) is maintained
        
        Args:
            message: User's message
            history: Optional conversation history (for non-session use)
            session_id: Optional session ID for multi-turn conversations
            user_id: User ID for preference learning (defaults to "default")
            
        Returns:
            ChatResponse with the assistant's reply and session_id
            
        Example:
            # Without session (stateless)
            response = bp.chat("What tools do I need for ChIP-seq analysis?")
            
            # With session (stateful)
            session = bp.session_manager.create_session("user_123")
            response = bp.chat("I work with mouse samples", 
                               session_id=session.session_id)
            # Context is maintained in follow-up
            response = bp.chat("Run RNA-seq", session_id=session.session_id)
        """
        user_id = user_id or "default"
        
        # Use professional ChatAgent for all chat interactions
        try:
            agent_response = self.chat_agent.process_message(
                content=message,
                session_id=session_id,
                user_id=user_id,
            )
            
            # Extract workflow if present in response metadata
            workflow = None
            if agent_response.entities_extracted and "workflow" in agent_response.entities_extracted:
                workflow = agent_response.entities_extracted["workflow"]
            
            return ChatResponse(
                message=agent_response.message.content,
                workflow=workflow,
                tools_used=[],  # Could be extracted from unified_result if present
                suggestions=agent_response.suggestions or [],
                session_id=agent_response.message.session_id,
            )
        except Exception as e:
            logger.error(f"ChatAgent error, falling back to UnifiedAgent: {e}")
            # Fallback to legacy path if ChatAgent fails
            return self._chat_legacy(message, history, session_id, user_id)
    
    def _chat_legacy(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """Legacy chat implementation using UnifiedAgent directly."""
        user_id = user_id or "default"
        
        # Prepare session (uses shared helper)
        session = self._prepare_chat_session(session_id, user_id, message)
        
        # Process message through agent (uses shared helper)
        result = self._process_agent_query(message)
        
        # Finalize session (uses shared helper)
        workflow = result.workflow if hasattr(result, "workflow") else None
        self._finalize_chat_session(session, result.message, workflow)
        
        return ChatResponse(
            message=result.message,
            workflow=workflow,
            tools_used=result.tools_used if hasattr(result, "tools_used") else [],
            suggestions=result.suggestions if hasattr(result, "suggestions") else [],
            session_id=session.session_id if session else None,
        )
    
    def chat_stream(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Streaming chat interface for real-time responses.
        
        Routes queries through the UnifiedAgent first to detect and execute tools,
        then yields the response. For queries that match tools (data scan, search,
        workflow generation), the tool is executed and results are yielded.
        For general queries, streams from the LLM.
        
        Args:
            message: User's message
            history: Optional conversation history
            session_id: Optional session ID for multi-turn conversations
            user_id: User ID for preference learning (defaults to "default")
            
        Yields:
            String chunks of the assistant's response
            
        Example:
            # Create a session
            session = bp.session_manager.create_session("user_123")
            
            # Stream with session context
            full_response = ""
            for chunk in bp.chat_stream("Explain RNA-seq", 
                                        session_id=session.session_id):
                print(chunk, end="", flush=True)
                full_response += chunk
        """
        user_id = user_id or "default"
        
        # Prepare session (uses shared helper)
        session = self._prepare_chat_session(session_id, user_id, message)
        
        try:
            # Detect what kind of query this is to show appropriate progress
            query_lower = message.lower()
            progress_message = None
            
            if any(word in query_lower for word in ['scan', 'list', 'find', 'what data', 'local data', 'local folder']):
                progress_message = "🔍 Scanning data directories...\n\n"
            elif any(word in query_lower for word in ['search', 'database', 'encode', 'geo', 'tcga']):
                progress_message = "🌐 Searching databases...\n\n"
            elif any(word in query_lower for word in ['generate', 'create', 'workflow', 'pipeline']):
                progress_message = "🧬 Generating workflow...\n\n"
            elif any(word in query_lower for word in ['submit', 'run', 'execute', 'launch']):
                progress_message = "🚀 Submitting job...\n\n"
            elif any(word in query_lower for word in ['status', 'job', 'running']):
                progress_message = "📊 Checking job status...\n\n"
            
            # Yield progress indicator immediately for tool-like queries
            if progress_message:
                yield progress_message
            
            # Process through the UnifiedAgent (uses shared helper)
            result = self._process_agent_query(message)
            
            # Check if a tool was executed (uses shared helper)
            tool_executed = self._was_tool_executed(result)
            
            if tool_executed or result.success:
                # Tool was executed or we got a good response - yield the result
                response_message = result.message or "Action completed."
                
                # Stream the response in chunks for smooth UX
                chunk_size = 50  # Characters per chunk for smooth streaming
                for i in range(0, len(response_message), chunk_size):
                    yield response_message[i:i + chunk_size]
                
                # Finalize session (uses shared helper)
                workflow = result.workflow if hasattr(result, "workflow") else None
                full_message = (progress_message or "") + response_message
                self._finalize_chat_session(session, full_message, workflow)
                return
            
            # No tool matched - fall back to LLM streaming for general queries
            from .providers import get_router
            router = get_router()
            
            # Get system prompt (uses shared helper)
            system_prompt = self._get_chat_system_prompt(session)
            
            # Collect full response for session storage
            full_response = []
            
            # Stream from the router
            for chunk in router.stream(message, system_prompt=system_prompt):
                full_response.append(chunk)
                yield chunk
            
            # Finalize session (uses shared helper)
            self._finalize_chat_session(session, "".join(full_response))
            
        except Exception as e:
            # Fallback to non-streaming
            logger.warning(f"Streaming failed, falling back: {e}")
            response = self.chat(message, history, session_id, user_id)
            yield response.message
    
    async def chat_stream_async(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Async streaming chat with true real-time progress for workflow generation.
        
        For workflow generation queries, this uses the multi-agent system which
        yields real progress updates as each phase (planning, codegen, validation,
        documentation) completes.
        
        Args:
            message: User's message
            history: Optional conversation history
            session_id: Optional session ID for multi-turn conversations
            user_id: User ID for preference learning (defaults to "default")
            
        Yields:
            String chunks of the assistant's response
            
        Example:
            async for chunk in bp.chat_stream_async("Generate RNA-seq workflow"):
                print(chunk, end="", flush=True)
        """
        user_id = user_id or "default"
        
        # Prepare session
        session = self._prepare_chat_session(session_id, user_id, message)
        
        # Check if this is a workflow generation query
        query_lower = message.lower()
        is_workflow_query = any(
            phrase in query_lower 
            for phrase in ['generate', 'create workflow', 'create pipeline', 'build workflow', 'make workflow']
        )
        
        if is_workflow_query:
            # Use multi-agent streaming for workflow generation
            try:
                yield "🧬 **Starting Workflow Generation**\n\n"
                
                async for update in self.generate_with_agents_streaming(message):
                    phase = update.get("phase", "")
                    status = update.get("status", "")
                    
                    if phase == "planning":
                        if status == "started":
                            yield "📋 **Planning**: Analyzing requirements...\n"
                        elif status == "complete":
                            plan_info = update.get("plan", {})
                            yield f"📋 **Planning**: Complete - {plan_info.get('steps', 0)} steps\n\n"
                    
                    elif phase == "codegen":
                        if status == "started":
                            yield "💻 **Code Generation**: Writing Nextflow code...\n"
                        elif status == "complete":
                            lines = update.get("lines", 0)
                            yield f"💻 **Code Generation**: Complete - {lines} lines\n\n"
                    
                    elif phase == "validation":
                        if status == "started":
                            yield "✅ **Validation**: Checking code...\n"
                        elif status == "complete":
                            issues = update.get("issues", 0)
                            yield f"✅ **Validation**: Complete - {issues} issues\n\n"
                    
                    elif phase == "documentation":
                        if status == "started":
                            yield "📚 **Documentation**: Generating README...\n"
                        elif status == "complete":
                            yield "📚 **Documentation**: Complete\n\n"
                    
                    elif phase == "complete":
                        result = update.get("result", {})
                        yield f"\n🎉 **Workflow Generated Successfully!**\n"
                        yield f"- Name: {result.get('name', 'Unknown')}\n"
                        yield f"- Lines of code: {result.get('code_lines', 0)}\n"
                        yield f"- Validation: {'Passed' if result.get('validation_passed') else 'Has issues'}\n"
                    
                    elif phase == "error":
                        yield f"\n❌ **Error**: {update.get('error', 'Unknown error')}\n"
                
                # Finalize session
                self._finalize_chat_session(session, "Workflow generation complete.")
                return
                
            except ImportError:
                # Multi-agent not available, fall back to sync
                yield "⚠️ Multi-agent streaming not available, using standard generation...\n\n"
        
        # For non-workflow queries, use sync processing (wrapped as async)
        for chunk in self.chat_stream(message, history, session_id, user_id):
            yield chunk
    
    def parse_intent(self, description: str) -> "ParsedIntent":
        """
        Parse a description to extract intent without generating workflow.
        
        Useful for understanding what the system detected from a query.
        
        Args:
            description: Natural language description
            
        Returns:
            ParsedIntent with analysis type, organism, etc.
        """
        self._ensure_initialized()
        return self._composer.parse_intent(description)
    
    # =========================================================================
    # Session Management API
    # =========================================================================
    
    def create_session(self, user_id: str = "default") -> str:
        """
        Create a new chat session.
        
        Sessions enable:
        - Multi-turn conversations with context
        - User preference learning
        - Conversation history persistence
        
        Args:
            user_id: User identifier for preference tracking
            
        Returns:
            Session ID to use in chat() calls
            
        Example:
            session_id = bp.create_session("user_123")
            response = bp.chat("I work with human samples", session_id=session_id)
            # Context is now tracked
            response = bp.chat("Run RNA-seq", session_id=session_id)
        """
        session = self.session_manager.create_session(user_id)
        return session.session_id
    
    def end_session(self, session_id: str) -> None:
        """
        End a session and persist its history.
        
        Args:
            session_id: Session to end
        """
        self.session_manager.end_session(session_id)
    
    def get_user_profile(self, user_id: str = "default") -> Dict[str, Any]:
        """
        Get a user's profile and preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with user preferences and history
            
        Example:
            profile = bp.get_user_profile("user_123")
            print(f"Preferred organism: {profile.get('preferred_organism')}")
        """
        from .agents.memory import get_profile_store
        store = get_profile_store()
        profile = store.load_profile(user_id)
        
        if profile:
            return profile.to_dict()
        return {"user_id": user_id, "message": "No profile found"}
    
    # =========================================================================
    # Core API - Data Discovery
    # =========================================================================
    
    def scan_data(
        self,
        path: Union[str, Path],
        pattern: str = "*.f*q*",
    ) -> "DataManifest":
        """
        Scan a directory for sequencing data files.
        
        Automatically detects:
        - Sample names from filenames
        - Paired-end vs single-end
        - File formats (FASTQ, FASTA, BAM, etc.)
        
        Args:
            path: Directory to scan
            pattern: File pattern to match
            
        Returns:
            DataManifest with discovered samples
            
        Example:
            manifest = bp.scan_data("/data/raw/project1")
            print(f"Found {len(manifest.samples)} samples")
        """
        from .data import LocalSampleScanner, DataManifest
        
        scanner = LocalSampleScanner()
        samples = scanner.scan_directory(str(path), pattern=pattern)
        
        return DataManifest(samples=samples)
    
    # =========================================================================
    # Core API - Job Execution
    # =========================================================================
    
    def submit(
        self,
        workflow: "Workflow",
        cluster: Optional[str] = None,
        **slurm_options
    ) -> JobStatus:
        """
        Submit a workflow for execution.
        
        Args:
            workflow: Workflow to execute
            cluster: Execution target ("local", "slurm")
            **slurm_options: SLURM-specific options
            
        Returns:
            JobStatus with job_id for tracking
            
        Example:
            job = bp.submit(workflow, cluster="slurm", partition="gpu")
            print(f"Submitted job {job.job_id}")
        """
        cluster = cluster or self._cluster
        
        if cluster == "local":
            return self._submit_local(workflow)
        elif cluster == "slurm":
            return self._submit_slurm(workflow, **slurm_options)
        else:
            raise ValueError(f"Unknown cluster type: {cluster}")
    
    def _submit_local(self, workflow: "Workflow") -> JobStatus:
        """Submit workflow for local execution."""
        # For now, just generate the workflow
        import uuid
        job_id = f"local-{uuid.uuid4().hex[:8]}"
        
        return JobStatus(
            job_id=job_id,
            state="pending",
            progress=0.0,
            message=f"Workflow ready at {workflow.output_dir}",
        )
    
    def _submit_slurm(self, workflow: "Workflow", **options) -> JobStatus:
        """Submit workflow to SLURM."""
        from .agents import AgentTools
        
        # Generate SLURM submission script
        script_path = workflow.output_dir / "submit.sh"
        
        # Use agent tools to submit
        result = AgentTools.slurm_submit(str(script_path), **options)
        
        if result.success:
            return JobStatus(
                job_id=result.output.get("job_id", "unknown"),
                state="pending",
                progress=0.0,
                message="Job submitted to SLURM",
            )
        else:
            return JobStatus(
                job_id="",
                state="failed",
                progress=0.0,
                message=f"Submission failed: {result.error}",
                errors=[result.error],
            )
    
    def status(self, job_id: str) -> JobStatus:
        """
        Get status of a submitted job.
        
        Args:
            job_id: Job ID from submit()
            
        Returns:
            JobStatus with current state
        """
        if job_id.startswith("local-"):
            return JobStatus(
                job_id=job_id,
                state="completed",
                progress=1.0,
                message="Local job completed",
            )
        
        # Query SLURM
        from .agents import AgentTools
        
        result = AgentTools.slurm_status(job_id)
        
        if result.success:
            slurm_state = result.output.get("state", "UNKNOWN")
            state_map = {
                "PENDING": "pending",
                "RUNNING": "running",
                "COMPLETED": "completed",
                "FAILED": "failed",
                "CANCELLED": "cancelled",
            }
            
            return JobStatus(
                job_id=job_id,
                state=state_map.get(slurm_state, "unknown"),
                progress=1.0 if slurm_state == "COMPLETED" else 0.5,
                message=result.output.get("message", ""),
            )
        else:
            return JobStatus(
                job_id=job_id,
                state="unknown",
                progress=0.0,
                message=f"Status query failed: {result.error}",
            )
    
    def cancel(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancellation succeeded
        """
        if job_id.startswith("local-"):
            return True
        
        from .agents import AgentTools
        
        result = AgentTools.slurm_cancel(job_id)
        return result.success
    
    # =========================================================================
    # Core API - Tool Discovery
    # =========================================================================
    
    def find_tools(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List["Tool"]:
        """
        Search the tool catalog.
        
        Args:
            query: Search query (tool name, description, etc.)
            category: Optional category filter
            limit: Maximum results
            
        Returns:
            List of matching tools
            
        Example:
            tools = bp.find_tools("alignment")
            for tool in tools:
                print(f"{tool.name}: {tool.description}")
        """
        self._ensure_initialized()
        
        return self._composer.tool_selector.search(
            query,
            category=category,
            limit=limit
        )
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        List available tools.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        self._ensure_initialized()
        
        if category:
            return self._composer.tool_selector.list_by_category(category)
        else:
            return self._composer.tool_selector.list_all()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def check_readiness(self, description: str) -> Dict[str, Any]:
        """
        Check if system is ready to generate a workflow.
        
        Validates:
        - Required tools are available
        - Container images exist
        - Modules are present
        
        Args:
            description: Workflow description
            
        Returns:
            Dict with readiness status and any issues
        """
        self._ensure_initialized()
        return self._composer.check_readiness(description)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available resources.
        
        Returns:
            Dict with tool count, module count, etc.
        """
        self._ensure_initialized()
        return self._composer.get_stats()
    
    def switch_llm(self, provider: str, model: Optional[str] = None) -> None:
        """
        Switch to a different LLM provider.
        
        Args:
            provider: Provider name ("lightning", "openai", etc.)
            model: Optional model name
        """
        self._ensure_initialized()
        self._composer.switch_llm(provider, model)
    
    # =========================================================================
    # Context Manager Support
    # =========================================================================
    
    def __enter__(self) -> "BioPipelines":
        """Support context manager usage."""
        return self
    
    def __exit__(self, *args) -> None:
        """Cleanup on exit."""
        pass
    
    def __repr__(self) -> str:
        provider = self._config.llm.default_provider if self._config else "unknown"
        return f"BioPipelines(llm={provider}, cluster={self._cluster})"


# =============================================================================
# Convenience Functions
# =============================================================================

_default_instance: Optional[BioPipelines] = None


def get_biopipelines(**kwargs) -> BioPipelines:
    """
    Get or create the default BioPipelines instance.
    
    For simple scripts that just need one instance.
    
    Example:
        bp = get_biopipelines()
        workflow = bp.generate("RNA-seq")
    """
    global _default_instance
    
    if _default_instance is None:
        _default_instance = BioPipelines(**kwargs)
    
    return _default_instance


def generate(description: str, **kwargs) -> "Workflow":
    """
    Quick function for workflow generation.
    
    Example:
        workflow = generate("RNA-seq for human samples")
    """
    return get_biopipelines().generate(description, **kwargs)


def chat(message: str) -> ChatResponse:
    """
    Quick function for chat interaction.
    
    Example:
        response = chat("What is RNA-seq?")
    """
    return get_biopipelines().chat(message)
