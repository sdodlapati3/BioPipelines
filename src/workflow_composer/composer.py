"""
Main Composer Class
===================

The main entry point for the AI Workflow Composer.

Orchestrates all components to:
1. Parse natural language intent
2. Select appropriate tools
3. Map to modules
4. Generate complete workflows

Example:
    from workflow_composer import Composer
    from workflow_composer.llm import get_llm
    
    composer = Composer(llm=get_llm("ollama"))
    workflow = composer.generate(
        "RNA-seq differential expression, mouse, paired-end"
    )
    workflow.save("my_workflow/")
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import Config
from .llm import LLMAdapter, get_llm
from .core import (
    IntentParser, ParsedIntent,
    ToolSelector, Tool,
    ModuleMapper, Module,
    WorkflowGenerator, Workflow
)

logger = logging.getLogger(__name__)


class Composer:
    """
    AI Workflow Composer - main orchestrator class.
    
    Takes natural language descriptions and generates complete
    Nextflow bioinformatics pipelines.
    """
    
    def __init__(
        self,
        llm: Optional[LLMAdapter] = None,
        config: Optional[Config] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the Workflow Composer.
        
        Args:
            llm: LLM adapter to use (optional, will use config default)
            config: Configuration object (optional)
            config_path: Path to config file (optional)
        """
        # Load configuration
        self.config = config or Config.load(config_path)
        
        # Set up LLM
        if llm:
            self.llm = llm
        else:
            provider = self.config.llm.default_provider
            model = self.config.get_llm_config(provider).model
            self.llm = get_llm(provider, model)
        
        # Initialize components
        self._init_components()
        
        logger.info(f"Composer initialized with LLM: {self.llm}")
    
    def _init_components(self) -> None:
        """Initialize all composer components."""
        base_path = self.config.base_path
        
        # Intent parser
        self.intent_parser = IntentParser(self.llm)
        
        # Tool selector
        catalog_path = self.config.resolve_path(
            self.config.knowledge_base.tool_catalog
        )
        self.tool_selector = ToolSelector(str(catalog_path))
        
        # Module mapper
        module_path = self.config.resolve_path(
            self.config.knowledge_base.module_library
        )
        self.module_mapper = ModuleMapper(str(module_path))
        
        # Workflow generator
        patterns_path = self.config.resolve_path(
            self.config.knowledge_base.workflow_patterns
        )
        self.workflow_generator = WorkflowGenerator(
            str(patterns_path) if patterns_path.exists() else None
        )
    
    def generate(
        self,
        description: str,
        output_dir: Optional[str] = None,
        auto_create_modules: bool = True,
        interactive: bool = False
    ) -> Workflow:
        """
        Generate a workflow from a natural language description.
        
        Args:
            description: Natural language description of the analysis
            output_dir: Directory to save workflow (optional)
            auto_create_modules: Auto-create missing modules using LLM
            interactive: Enable interactive clarification
            
        Returns:
            Generated Workflow object
        """
        logger.info(f"Generating workflow from: {description[:100]}...")
        
        # Step 1: Parse intent
        logger.info("Step 1: Parsing intent...")
        intent = self.intent_parser.parse(description)
        logger.info(f"  Analysis type: {intent.analysis_type.value}")
        logger.info(f"  Organism: {intent.organism}")
        logger.info(f"  Confidence: {intent.confidence:.2f}")
        
        # Interactive clarification if needed
        if interactive and intent.confidence < 0.7:
            logger.info("  Low confidence - would prompt for clarification")
            # TODO: Implement interactive mode
        
        # Step 2: Select tools
        logger.info("Step 2: Selecting tools...")
        tool_map = self.tool_selector.find_tools_for_analysis(
            intent.analysis_type.value
        )
        
        # Flatten tool list
        all_tools = []
        for category, tools in tool_map.items():
            logger.info(f"  {category}: {[t.name for t in tools]}")
            all_tools.extend(tools)
        
        # Step 3: Map to modules
        logger.info("Step 3: Mapping to modules...")
        tool_names = [t.name for t in all_tools]
        module_map = self.module_mapper.find_modules_for_tools(tool_names)
        
        modules = []
        missing = []
        for tool_name, module in module_map.items():
            if module:
                modules.append(module)
                logger.info(f"  ✓ {tool_name} -> {module.name}")
            else:
                missing.append(tool_name)
                logger.warning(f"  ✗ {tool_name} -> module not found")
        
        # Auto-create missing modules
        if missing and auto_create_modules:
            logger.info(f"Creating {len(missing)} missing modules...")
            for tool_name in missing:
                tool = self.tool_selector.find_tool(tool_name)
                if tool:
                    container = tool.container
                else:
                    container = "base"
                
                try:
                    module = self.module_mapper.create_module(
                        tool_name, container, self.llm
                    )
                    modules.append(module)
                    logger.info(f"  Created: {module.name}")
                except Exception as e:
                    logger.error(f"  Failed to create {tool_name}: {e}")
        
        # Step 4: Generate workflow
        logger.info("Step 4: Generating workflow...")
        workflow = self.workflow_generator.generate(
            intent, modules, self.llm
        )
        
        # Save if output_dir specified
        if output_dir:
            workflow.save(output_dir)
        
        logger.info(f"Workflow generation complete: {workflow.name}")
        return workflow
    
    def parse_intent(self, description: str) -> ParsedIntent:
        """
        Parse a description to extract intent (without generating workflow).
        
        Args:
            description: Natural language description
            
        Returns:
            ParsedIntent object
        """
        return self.intent_parser.parse(description)
    
    def find_tools(self, analysis_type: str) -> Dict[str, List[Tool]]:
        """
        Find tools for an analysis type.
        
        Args:
            analysis_type: Analysis type string
            
        Returns:
            Dict mapping categories to tool lists
        """
        return self.tool_selector.find_tools_for_analysis(analysis_type)
    
    def find_modules(self, tool_names: List[str]) -> Dict[str, Optional[Module]]:
        """
        Find modules for a list of tools.
        
        Args:
            tool_names: List of tool names
            
        Returns:
            Dict mapping tool names to modules (or None)
        """
        return self.module_mapper.find_modules_for_tools(tool_names)
    
    def check_readiness(self, description: str) -> Dict[str, Any]:
        """
        Check if all components are ready to generate a workflow.
        
        Args:
            description: Natural language description
            
        Returns:
            Dict with readiness status and any issues
        """
        result = {
            "ready": True,
            "issues": [],
            "warnings": []
        }
        
        # Parse intent
        try:
            intent = self.intent_parser.parse(description)
            result["intent"] = intent.to_dict()
        except Exception as e:
            result["ready"] = False
            result["issues"].append(f"Intent parsing failed: {e}")
            return result
        
        if intent.confidence < 0.5:
            result["warnings"].append(
                f"Low confidence ({intent.confidence:.2f}) in intent parsing"
            )
        
        # Find tools
        tool_map = self.tool_selector.find_tools_for_analysis(
            intent.analysis_type.value
        )
        
        all_tools = []
        for tools in tool_map.values():
            all_tools.extend(tools)
        
        if not all_tools:
            result["ready"] = False
            result["issues"].append(
                f"No tools found for analysis type: {intent.analysis_type.value}"
            )
        
        result["tools_found"] = len(all_tools)
        
        # Find modules
        tool_names = [t.name for t in all_tools]
        module_map = self.module_mapper.find_modules_for_tools(tool_names)
        
        missing = [name for name, mod in module_map.items() if mod is None]
        
        if missing:
            result["warnings"].append(
                f"Missing modules for: {', '.join(missing)}"
            )
        
        result["modules_found"] = sum(1 for m in module_map.values() if m)
        result["modules_missing"] = missing
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about available resources."""
        return {
            "llm": str(self.llm),
            "tool_catalog": self.tool_selector.get_stats(),
            "modules": {
                "total": len(self.module_mapper.modules),
                "by_category": self.module_mapper.list_by_category()
            }
        }
    
    def switch_llm(self, provider: str, model: Optional[str] = None) -> None:
        """
        Switch to a different LLM provider.
        
        Args:
            provider: Provider name (ollama, openai, anthropic, huggingface)
            model: Optional model name
        """
        self.llm = get_llm(provider, model)
        self.intent_parser = IntentParser(self.llm)
        logger.info(f"Switched to LLM: {self.llm}")
