"""
Archived Agent Components
=========================

These modules have been superseded by the unified agent architecture.

MIGRATION GUIDE:
---------------

AgentBridge (bridge.py):
    BEFORE:
        from workflow_composer.agents import AgentBridge
        bridge = AgentBridge(app_state)
        result = await bridge.process_message(message, context)
    
    AFTER:
        from workflow_composer.agents import UnifiedAgent
        agent = UnifiedAgent()
        response = agent.process_sync(message)

AgentRouter (router.py):
    BEFORE:
        from workflow_composer.agents import AgentRouter
        router = AgentRouter()
        result = await router.route(message)
    
    AFTER:
        from workflow_composer.agents import UnifiedAgent
        agent = UnifiedAgent()
        # Uses HybridQueryParser internally for intent detection
        response = agent.process_sync(message)

REASONING:
---------
- AgentBridge was a transitional layer between LLM routing and tools
- AgentRouter was LLM-based intent detection, now replaced by HybridQueryParser
- UnifiedAgent integrates all functionality with proper permission control

DATE ARCHIVED: 2025-11-29
"""

# Backward compatibility imports (emit deprecation warnings)
import warnings

def _get_bridge():
    warnings.warn(
        "AgentBridge is deprecated and archived. Use UnifiedAgent instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from ._archived.bridge import AgentBridge
    return AgentBridge

def _get_router():
    warnings.warn(
        "AgentRouter is deprecated and archived. Use UnifiedAgent with HybridQueryParser instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from ._archived.router import AgentRouter
    return AgentRouter
