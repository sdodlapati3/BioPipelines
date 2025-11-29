# Architecture Diagrams

This directory contains visual architecture diagrams for the BioPipelines system.

## Diagram Index

| File | Description | Format |
|------|-------------|--------|
| `system_overview.mmd` | High-level layered system architecture | Mermaid |
| `unified_agent_flow.mmd` | UnifiedAgent query processing flowchart | Mermaid |
| `tool_execution.mmd` | Tool execution state machine with permissions | Mermaid |
| `data_flow.mmd` | Complete request/response sequence diagram | Mermaid |
| `component_relationships.mmd` | Module dependencies graph | Mermaid |
| `permission_state.mmd` | Autonomy level state transitions | Mermaid |
| `llm_provider_sequence.mmd` | LLM provider routing sequence | Mermaid |
| `workflow_generation.mmd` | Workflow generation pipeline | Mermaid |
| `autonomous_loop.mmd` | Autonomous monitoring and recovery loop | Mermaid |
| `tool_catalog.mmd` | All 32 tools with permission mappings | Mermaid |
| `disconnected_components.mmd` | Analysis of isolated/redundant modules | Mermaid |

## Viewing Diagrams

### Option 1: VS Code Extension
Install "Markdown Preview Mermaid Support" extension.

### Option 2: Online Viewer
Paste content into https://mermaid.live

### Option 3: Generate Images
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate PNG
mmdc -i system_overview.mmd -o system_overview.png

# Generate SVG
mmdc -i system_overview.mmd -o system_overview.svg
```

## Quick Preview

All diagrams use Mermaid syntax. See individual `.mmd` files for source.
