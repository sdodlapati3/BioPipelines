# BioPipelines MCP Integration Guide

## Overview

BioPipelines provides a Model Context Protocol (MCP) server that exposes its bioinformatics tools to MCP-compatible clients like Claude Code, Cursor, and other AI coding assistants.

## Quick Start

### Running the MCP Server

```bash
# Run using stdio transport (default, for Claude Code)
python -m workflow_composer.mcp.server

# Run using HTTP transport (for development/testing)
python -m workflow_composer.mcp.server --transport http --port 8080
```

### Configuration for Claude Code

Add to your `mcp-config.json` or Claude Code settings:

```json
{
  "mcpServers": {
    "biopipelines": {
      "command": "python",
      "args": ["-m", "workflow_composer.mcp.server"],
      "env": {
        "PYTHONPATH": "/path/to/BioPipelines/src:/path/to/BioPipelines/config"
      }
    }
  }
}
```

### Configuration for Cursor

Add to your Cursor MCP settings:

```json
{
  "mcp": {
    "servers": {
      "biopipelines": {
        "command": "python",
        "args": ["-m", "workflow_composer.mcp.server"],
        "cwd": "/path/to/BioPipelines"
      }
    }
  }
}
```

## Available Tools

### Data Discovery

#### `search_encode`
Search the ENCODE database for chromatin accessibility, histone modifications, and gene expression data.

**Parameters:**
- `query` (string, required): Search terms
- `assay_type` (string): Filter by assay (ChIP-seq, ATAC-seq, RNA-seq, etc.)
- `organism` (string): Filter by organism (human, mouse)
- `limit` (integer): Maximum results (default: 10)

**Example:**
```
Search ENCODE for K562 ChIP-seq H3K27ac data
```

#### `search_geo`
Search NCBI GEO database for gene expression datasets.

**Parameters:**
- `query` (string, required): Search terms
- `organism` (string): Filter by organism
- `limit` (integer): Maximum results (default: 10)

### Workflow Generation

#### `create_workflow`
Generate a bioinformatics analysis workflow.

**Parameters:**
- `analysis_type` (string, required): rnaseq, chipseq, methylation, variant, atacseq
- `organism` (string): Reference organism (default: human)
- `input_dir` (string): Input data directory
- `output_dir` (string): Output directory
- `workflow_engine` (string): nextflow or snakemake (default: nextflow)

**Example:**
```
Create an RNA-seq workflow for mouse samples in /data/rnaseq/
```

#### `use_workflow_template`
Generate a workflow from a pre-built template.

**Parameters:**
- `template_name` (string, required): Template name (e.g., basic_de, full_analysis)
- `parameters` (object): Template parameters
- `output_dir` (string): Output directory

### Database Queries

#### `search_uniprot`
Search UniProt for protein information.

**Parameters:**
- `query` (string, required): Gene or protein name
- `organism` (string): Filter by organism (default: human)
- `reviewed` (boolean): Only Swiss-Prot entries (default: true)
- `limit` (integer): Maximum results (default: 25)

**Example:**
```
Search UniProt for TP53 in human
```

#### `get_protein_interactions`
Get protein-protein interactions from STRING database.

**Parameters:**
- `genes` (array, required): List of gene names
- `organism` (string): Organism (default: human)
- `score_threshold` (integer): Minimum score (default: 400)

**Example:**
```
Get interactions for BRCA1, BRCA2, and TP53
```

#### `get_functional_enrichment`
Run Gene Ontology and pathway enrichment analysis.

**Parameters:**
- `genes` (array, required): List of gene names
- `organism` (string): Organism (default: human)

**Example:**
```
Get enrichment for differentially expressed genes: ["EGFR", "KRAS", "MYC", "TP53"]
```

#### `search_kegg_pathways`
Search KEGG for metabolic and signaling pathways.

**Parameters:**
- `query` (string, required): Pathway name or keywords
- `organism` (string): KEGG organism code (default: hsa)

#### `search_pubmed`
Search PubMed for scientific literature.

**Parameters:**
- `query` (string, required): Search terms
- `limit` (integer): Maximum results (default: 10)
- `sort` (string): relevance or date (default: relevance)

#### `search_variants`
Search ClinVar for variant pathogenicity.

**Parameters:**
- `gene` (string, required): Gene symbol
- `significance` (string): Filter by clinical significance
- `limit` (integer): Maximum results (default: 25)

### Educational Tools

#### `explain_concept`
Get detailed explanations of bioinformatics concepts.

**Parameters:**
- `concept` (string, required): The concept to explain
- `level` (string): beginner, intermediate, advanced (default: intermediate)

**Example:**
```
Explain DESeq2 for beginners
```

### Job Management

#### `check_job_status`
Check the status of a submitted workflow job.

**Parameters:**
- `job_id` (string, required): Job identifier

## Available Resources

### `biopipelines://skills`
List of all available BioPipelines skills and their capabilities.

### `biopipelines://templates`
List of available pre-built workflow templates with descriptions.

### `biopipelines://databases`
List of integrated biological databases and their features.

## Example Workflows

### RNA-seq Analysis

```
1. Search GEO for relevant datasets
2. Create an RNA-seq workflow for the data
3. Get functional enrichment for differentially expressed genes
4. Search PubMed for related publications
```

### Protein Analysis

```
1. Search UniProt for proteins of interest
2. Get protein-protein interactions
3. Run pathway enrichment analysis
4. Visualize results
```

### Variant Analysis

```
1. Create a variant calling workflow
2. Search ClinVar for variant annotations
3. Get pathway enrichment for affected genes
```

## Development

### Testing

```bash
# Run MCP server tests
python -m pytest tests/test_mcp_server.py -v
```

### Adding New Tools

To add a new tool, register it in `_setup_tools()`:

```python
self._register_tool(
    name="my_new_tool",
    description="Description of what the tool does",
    parameters={
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param1"]
    },
    handler=self._handle_my_new_tool
)
```

Then implement the handler:

```python
async def _handle_my_new_tool(self, **kwargs) -> Dict[str, Any]:
    try:
        # Tool implementation
        result = do_something(kwargs.get("param1"))
        return {
            "success": True,
            "content": format_result(result)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Troubleshooting

### Server Not Starting

1. Ensure Python environment is activated
2. Check PYTHONPATH includes src/ and config/ directories
3. Verify all dependencies are installed

### Tools Not Working

1. Check that database clients are properly configured
2. Verify network access to external APIs
3. Check rate limiting isn't causing issues

### Connection Issues

1. For stdio transport, ensure proper JSON communication
2. For HTTP transport, check port availability
3. Verify firewall allows connections

## Protocol Version

This server implements MCP protocol version `2024-11-05`.

## License

MIT License - See LICENSE file for details.
