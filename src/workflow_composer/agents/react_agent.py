"""
ReAct Agent
============

Multi-step ReAct (Reason + Act) agent for complex tasks.

Implements the ReAct pattern:
1. Thought - What should I do?
2. Action - Which tool to use?
3. Observation - What happened?
4. Repeat until done

Enhanced with DeepCode-inspired patterns:
- JSON repair for parsing action inputs
- Response validation before parsing
- Token tracking for long workflows
- Concise memory for multi-step tasks

References:
- ReAct: Synergizing Reasoning and Acting in LLMs (https://arxiv.org/abs/2210.03629)
- DeepCode: workflows/agents/memory_agent_concise.py
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, AsyncGenerator, Callable
from enum import Enum
from datetime import datetime

# DeepCode-inspired utilities
from .utils.json_repair import safe_json_loads, extract_json_from_text
from .utils.response_validator import ResponseValidator, ValidationResult
from .memory.token_tracker import TokenTracker, TokenBudget, create_tracker_for_model
from .memory.concise_memory import ConciseMemory, create_concise_memory

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class AgentState(Enum):
    """Current state of the agent."""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    RESPONDING = "responding"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain."""
    step_num: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    state: AgentState = AgentState.THINKING
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_prompt(self) -> str:
        """Convert to prompt format."""
        parts = [f"Step {self.step_num}:"]
        parts.append(f"Thought: {self.thought}")
        if self.action:
            parts.append(f"Action: {self.action}")
            if self.action_input:
                parts.append(f"Action Input: {self.action_input}")
        if self.observation:
            parts.append(f"Observation: {self.observation}")
        return "\n".join(parts)


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    tool_name: str
    output: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# =============================================================================
# ReAct Agent
# =============================================================================

REACT_SYSTEM_PROMPT = """You are a ReAct agent for bioinformatics workflows. You solve problems step by step.

For each step, you must follow this format:
Thought: <your reasoning about what to do next>
Action: <tool name to use, or "finish" if done>
Action Input: <JSON input for the tool>

Available tools:
{tools}

Important:
- Think carefully before each action
- Use observations to inform next steps
- Say "Action: finish" when you have the final answer
- Be concise but thorough
"""


class ReactAgent:
    """
    ReAct-style agent for multi-step reasoning and action.
    
    Flow:
    1. User query → Thought (what should I do?)
    2. Thought → Action (which tool to use?)
    3. Action → Observation (what happened?)
    4. Observation → Thought (what next?)
    5. Repeat until done or max_steps
    6. Synthesize final response
    
    Enhanced with DeepCode patterns:
    - Token tracking to prevent context overflow
    - JSON repair for robust action input parsing
    - Response validation before parsing
    - Concise memory for long multi-step workflows
    
    Example:
        agent = ReactAgent(tools=my_tools, llm_client=client)
        response = await agent.run("Scan /data/fastq for RNA-seq samples")
    """
    
    def __init__(
        self,
        tools: Dict[str, Callable],
        llm_client: Any,  # OpenAI-compatible client
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        max_steps: int = 5,
        verbose: bool = True,
        use_concise_memory: bool = False,  # Enable for long workflows
        token_budget: Optional[TokenBudget] = None,
    ):
        """
        Initialize ReAct agent.
        
        Args:
            tools: Dictionary of tool name -> callable
            llm_client: OpenAI-compatible client (vLLM or API)
            model: Model name for completions
            max_steps: Maximum reasoning steps
            verbose: Log intermediate steps
            use_concise_memory: Use concise memory pattern for long workflows
            token_budget: Token budget configuration (auto-detected if None)
        """
        self.tools = tools
        self.client = llm_client
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        
        self.steps: List[AgentStep] = []
        self.state = AgentState.THINKING
        
        # Token tracking (DeepCode pattern)
        self.token_tracker = create_tracker_for_model(model)
        if token_budget:
            self.token_tracker.budget = token_budget
        
        # Concise memory for long workflows (DeepCode pattern)
        self.use_concise_memory = use_concise_memory
        self._concise_memory: Optional[ConciseMemory] = None
        
        # Track JSON repair statistics
        self._json_repair_count = 0
        self._validation_warnings: List[str] = []
    
    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Run the agent on a user query.
        
        Args:
            query: User's natural language query
            context: Additional context (data loaded, active jobs, etc.)
            stream_callback: Optional callback for streaming updates
            
        Returns:
            Final response to the user
        """
        self.steps = []
        self.state = AgentState.THINKING
        self._json_repair_count = 0
        self._validation_warnings = []
        
        # Initialize concise memory if enabled
        if self.use_concise_memory:
            system_prompt = self._build_system_prompt()
            self._concise_memory = create_concise_memory(
                system_prompt=system_prompt,
                model_name=self.model,
            )
            self._concise_memory.set_initial_query(query)
        
        # Build initial prompt
        system_prompt = self._build_system_prompt()
        
        # Track tokens for system prompt
        self.token_tracker.add_system_prompt(system_prompt)
        
        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                logger.info(f"ReAct Step {step_num}/{self.max_steps}")
                # Log token status
                token_status = self.token_tracker.get_status()
                logger.debug(f"Token usage: {token_status['usage']['total']}/{token_status['budget']['max_context']}")
            
            if stream_callback:
                stream_callback(f"🤔 Step {step_num}: Thinking...\n")
            
            # Get next thought/action from LLM
            messages = self._build_messages(query, context)
            
            # Track message tokens
            for msg in messages:
                self.token_tracker.add_message(msg["role"], msg.get("content", ""))
            
            try:
                response = await self._get_completion(system_prompt, messages)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return f"Error communicating with LLM: {e}"
            
            # Parse response
            thought, action, action_input = self._parse_response(response)
            
            step = AgentStep(
                step_num=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                state=AgentState.ACTING if action else AgentState.THINKING
            )
            
            if self.verbose:
                logger.info(f"Thought: {thought[:100]}...")
                if action:
                    logger.info(f"Action: {action}")
            
            if stream_callback:
                stream_callback(f"💭 {thought}\n")
            
            # Check if done
            if action and action.lower() == "finish":
                step.state = AgentState.DONE
                self.steps.append(step)
                
                # Complete step in concise memory
                if self._concise_memory:
                    self._concise_memory.complete_step(
                        step_num=step_num,
                        action="finish",
                        summary="Task completed successfully",
                    )
                
                # Extract final answer from action_input or thought
                final_answer = self._extract_final_answer(action_input, thought)
                if stream_callback:
                    stream_callback(f"✅ Done!\n\n")
                return final_answer
            
            # Execute action
            if action and action in self.tools:
                if stream_callback:
                    stream_callback(f"⚡ Executing: {action}...\n")
                
                step.state = AgentState.ACTING
                result = await self._execute_tool(action, action_input or {})
                
                step.observation = result.output if result.success else f"Error: {result.error}"
                step.state = AgentState.OBSERVING
                
                # Track tool result tokens
                self.token_tracker.add_tool_result(step.observation)
                
                # Update concise memory if enabled
                if self._concise_memory:
                    self._concise_memory.add_working_message("assistant", f"Action: {action}")
                    self._concise_memory.add_tool_result(step.observation, action)
                
                if stream_callback:
                    obs_preview = step.observation[:200] + "..." if len(step.observation) > 200 else step.observation
                    stream_callback(f"📋 Result: {obs_preview}\n\n")
            
            elif action:
                step.observation = f"Unknown tool: {action}. Available: {list(self.tools.keys())}"
            
            self.steps.append(step)
            
            # Check if we should compress context (DeepCode pattern)
            if self.token_tracker.should_compress():
                logger.warning(f"Token budget exceeded ({self.token_tracker.get_usage_percentage():.1f}%), consider using concise_memory=True")
        
        # Max steps reached - synthesize response
        if stream_callback:
            stream_callback("⚠️ Max steps reached, synthesizing response...\n")
        
        return self._synthesize_response(query)
    
    async def run_streaming(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Run agent with streaming output.
        
        Yields intermediate updates and final response.
        """
        buffer = []
        
        def callback(text: str):
            buffer.append(text)
        
        # Run agent with callback
        final = await self.run(query, context, stream_callback=callback)
        
        # Yield buffered updates
        for text in buffer:
            yield text
        
        # Yield final response
        yield final
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with available tools."""
        tools_desc = []
        for name, func in self.tools.items():
            doc = func.__doc__ or "No description"
            # Get first line of docstring
            first_line = doc.strip().split('\n')[0]
            tools_desc.append(f"- {name}: {first_line}")
        
        return REACT_SYSTEM_PROMPT.format(tools="\n".join(tools_desc))
    
    def _build_messages(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Build messages for LLM."""
        messages = []
        
        # Add context if available
        if context:
            context_str = self._format_context(context)
            messages.append({
                "role": "user",
                "content": f"Current context:\n{context_str}\n\nUser query: {query}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"User query: {query}"
            })
        
        # Add previous steps
        if self.steps:
            steps_str = "\n\n".join(step.to_prompt() for step in self.steps)
            messages.append({
                "role": "assistant",
                "content": steps_str
            })
            messages.append({
                "role": "user",
                "content": "Continue with the next step:"
            })
        
        return messages
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        parts = []
        if context.get("data_loaded"):
            parts.append(f"- Data loaded: {context.get('sample_count', 0)} samples from {context.get('data_path', 'unknown')}")
        if context.get("last_workflow"):
            parts.append(f"- Last workflow: {context['last_workflow']}")
        if context.get("active_job"):
            parts.append(f"- Active job: {context['active_job']}")
        return "\n".join(parts) if parts else "No active context"
    
    async def _get_completion(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> str:
        """Get completion from LLM."""
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Handle both sync and async clients
        if asyncio.iscoroutinefunction(getattr(self.client.chat.completions, 'create', None)):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=0.7,
                max_tokens=1024,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=0.7,
                max_tokens=1024,
            )
        
        return response.choices[0].message.content
    
    def _parse_response(self, response: str) -> tuple:
        """
        Parse LLM response into thought, action, action_input.
        
        Uses JSON repair for robust action input parsing (DeepCode pattern).
        Validates response structure before parsing.
        """
        thought = ""
        action = None
        action_input = None
        
        # Validate response structure first
        validation = ResponseValidator.validate_react_response(response)
        if validation.warnings:
            self._validation_warnings.extend(validation.warnings)
            if self.verbose:
                for warning in validation.warnings:
                    logger.debug(f"Response validation warning: {warning}")
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Action Input:"):
                input_str = line[13:].strip()
                
                # Use JSON repair for robust parsing (DeepCode pattern)
                if input_str:
                    # First try direct parse
                    parsed = safe_json_loads(input_str, default=None)
                    if parsed is not None:
                        action_input = parsed
                    else:
                        # Try extracting JSON from surrounding text
                        extracted = extract_json_from_text(input_str)
                        if extracted:
                            action_input = safe_json_loads(extracted, default={"raw": input_str})
                            self._json_repair_count += 1
                            if self.verbose:
                                logger.debug(f"Repaired JSON action input (repair #{self._json_repair_count})")
                        else:
                            # Fall back to raw string
                            action_input = {"raw": input_str}
        
        # If no structured thought, use entire response up to Action:
        if not thought:
            thought = response.split("Action:")[0].strip()
        
        return thought, action, action_input
    
    async def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool and return result."""
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                output="",
                error=f"Unknown tool: {tool_name}"
            )
        
        try:
            # Handle both sync and async tools
            if asyncio.iscoroutinefunction(tool):
                result = await tool(**args)
            else:
                result = tool(**args)
            
            # Handle different result types
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, dict):
                return ToolResult(
                    success=True,
                    tool_name=tool_name,
                    output=str(result.get("message", result)),
                    data=result
                )
            else:
                return ToolResult(
                    success=True,
                    tool_name=tool_name,
                    output=str(result),
                )
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                output="",
                error=str(e)
            )
    
    def _extract_final_answer(
        self,
        action_input: Optional[Dict],
        thought: str,
    ) -> str:
        """Extract final answer from finish action."""
        if action_input:
            if isinstance(action_input, dict):
                return action_input.get("answer", action_input.get("response", str(action_input)))
            return str(action_input)
        return thought
    
    def _synthesize_response(self, query: str) -> str:
        """Synthesize response from all steps when max steps reached."""
        if not self.steps:
            return "I wasn't able to complete your request. Could you provide more details?"
        
        # Find the most useful observation
        for step in reversed(self.steps):
            if step.observation and not step.observation.startswith("Error"):
                return f"Based on my analysis:\n\n{step.observation}"
        
        # Fall back to last thought
        last_step = self.steps[-1]
        return f"I was working on your request but couldn't complete all steps. Here's what I found:\n\n{last_step.thought}"
    
    def get_trace(self) -> str:
        """Get full reasoning trace for debugging."""
        if not self.steps:
            return "No steps recorded"
        
        trace_parts = ["=== ReAct Trace ===\n"]
        for step in self.steps:
            trace_parts.append(step.to_prompt())
            trace_parts.append("")
        
        return "\n".join(trace_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent execution statistics.
        
        Includes token usage, JSON repair stats, and validation info.
        Useful for debugging and optimization.
        """
        stats = {
            "steps_executed": len(self.steps),
            "max_steps": self.max_steps,
            "state": self.state.value,
            "model": self.model,
            "token_usage": self.token_tracker.get_status(),
            "json_repairs": self._json_repair_count,
            "validation_warnings": self._validation_warnings,
            "use_concise_memory": self.use_concise_memory,
        }
        
        if self._concise_memory:
            stats["concise_memory"] = self._concise_memory.get_stats()
        
        return stats
    
    def get_token_status(self) -> Dict[str, Any]:
        """Get current token usage status."""
        return self.token_tracker.get_status()


# =============================================================================
# Simple Agent (Single-Step)
# =============================================================================

class SimpleAgent:
    """
    Simple single-step agent for straightforward queries.
    
    Unlike ReAct, this just:
    1. Parses the query for tool calls
    2. Executes one tool
    3. Returns the result
    
    Use for simple commands like "scan /data" or "search ENCODE for ChIP-seq".
    """
    
    def __init__(
        self,
        tools: Dict[str, Callable],
        llm_client: Any = None,
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
    ):
        self.tools = tools
        self.client = llm_client
        self.model = model
    
    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run simple one-shot query."""
        # Try pattern matching first
        tool_name, args = self._pattern_match(query)
        
        if tool_name and tool_name in self.tools:
            result = await self._execute(tool_name, args)
            return result.output if result.success else f"Error: {result.error}"
        
        # Fall back to LLM if available
        if self.client:
            return await self._llm_route(query, context)
        
        return "I couldn't understand your request. Try: 'scan /path/to/data' or 'search ENCODE for ChIP-seq'"
    
    def _pattern_match(self, query: str) -> tuple:
        """Simple pattern matching for common queries."""
        query_lower = query.lower()
        
        # Scan data
        if "scan" in query_lower:
            import re
            path_match = re.search(r'scan\s+(\S+)', query, re.IGNORECASE)
            if path_match:
                return "scan_data", {"path": path_match.group(1)}
        
        # Search databases
        if "search" in query_lower:
            if "encode" in query_lower:
                return "search_encode", {"query": query}
            if "geo" in query_lower or "sra" in query_lower:
                return "search_geo", {"query": query}
        
        # Generate workflow
        if "workflow" in query_lower or "pipeline" in query_lower:
            return "generate_workflow", {"description": query}
        
        return None, {}
    
    async def _execute(self, tool_name: str, args: Dict) -> ToolResult:
        """Execute a tool."""
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(success=False, tool_name=tool_name, output="", error="Unknown tool")
        
        try:
            if asyncio.iscoroutinefunction(tool):
                result = await tool(**args)
            else:
                result = tool(**args)
            
            if isinstance(result, ToolResult):
                return result
            return ToolResult(success=True, tool_name=tool_name, output=str(result))
        except Exception as e:
            return ToolResult(success=False, tool_name=tool_name, output="", error=str(e))
    
    async def _llm_route(self, query: str, context: Optional[Dict]) -> str:
        """Use LLM for routing."""
        # This would use the router for more complex queries
        return "LLM routing not implemented in SimpleAgent"
