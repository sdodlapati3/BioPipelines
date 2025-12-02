"""
Tests for BioPipelines MCP Server
==================================

Tests for the Model Context Protocol server implementation.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflow_composer.mcp.server import (
    BioPipelinesMCPServer,
    create_server,
    ToolDefinition,
    ResourceDefinition,
)


class TestMCPServerInitialization:
    """Tests for MCP server initialization."""
    
    def test_create_server(self):
        """Test server creation."""
        server = create_server()
        assert isinstance(server, BioPipelinesMCPServer)
    
    def test_server_has_tools(self):
        """Test that server has registered tools."""
        server = create_server()
        assert len(server.tools) > 0
    
    def test_server_has_resources(self):
        """Test that server has registered resources."""
        server = create_server()
        assert len(server.resources) > 0
    
    def test_tools_have_required_fields(self):
        """Test that all tools have required fields."""
        server = create_server()
        
        for name, tool in server.tools.items():
            assert tool.name == name
            assert tool.description
            assert "type" in tool.parameters
            assert callable(tool.handler)
    
    def test_resources_have_required_fields(self):
        """Test that all resources have required fields."""
        server = create_server()
        
        for uri, resource in server.resources.items():
            assert resource.uri == uri
            assert resource.name
            assert resource.description
            assert callable(resource.handler)


class TestMCPToolList:
    """Tests for tool listing functionality."""
    
    def test_get_tools_list_format(self):
        """Test tools list format."""
        server = create_server()
        tools = server.get_tools_list()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
    
    def test_expected_tools_present(self):
        """Test that expected tools are registered."""
        server = create_server()
        tool_names = [t["name"] for t in server.get_tools_list()]
        
        expected_tools = [
            "search_encode",
            "search_geo",
            "create_workflow",
            "search_uniprot",
            "get_protein_interactions",
            "get_functional_enrichment",
            "search_kegg_pathways",
            "search_pubmed",
            "search_variants",
            "explain_concept",
        ]
        
        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"


class TestMCPResourceList:
    """Tests for resource listing functionality."""
    
    def test_get_resources_list_format(self):
        """Test resources list format."""
        server = create_server()
        resources = server.get_resources_list()
        
        assert isinstance(resources, list)
        assert len(resources) > 0
        
        for resource in resources:
            assert "uri" in resource
            assert "name" in resource
            assert "description" in resource
    
    def test_expected_resources_present(self):
        """Test that expected resources are registered."""
        server = create_server()
        resource_uris = [r["uri"] for r in server.get_resources_list()]
        
        expected_resources = [
            "biopipelines://skills",
            "biopipelines://templates",
            "biopipelines://databases",
        ]
        
        for expected in expected_resources:
            assert expected in resource_uris, f"Missing resource: {expected}"


class TestMCPProtocol:
    """Tests for MCP protocol handling."""
    
    @pytest.fixture
    def server(self):
        """Create server fixture."""
        return create_server()
    
    @pytest.mark.asyncio
    async def test_initialize_request(self, server):
        """Test initialize protocol method."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "protocolVersion" in response["result"]
        assert "serverInfo" in response["result"]
        assert response["result"]["serverInfo"]["name"] == "biopipelines"
    
    @pytest.mark.asyncio
    async def test_tools_list_request(self, server):
        """Test tools/list protocol method."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0
    
    @pytest.mark.asyncio
    async def test_resources_list_request(self, server):
        """Test resources/list protocol method."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert "resources" in response["result"]
    
    @pytest.mark.asyncio
    async def test_unknown_method(self, server):
        """Test handling of unknown method."""
        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "unknown/method",
            "params": {}
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "error" in response
        assert response["error"]["code"] == -32601


class TestMCPToolCalls:
    """Tests for MCP tool call handling."""
    
    @pytest.fixture
    def server(self):
        """Create server fixture."""
        return create_server()
    
    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, server):
        """Test calling an unknown tool."""
        result = await server.call_tool("unknown_tool", {})
        
        assert not result["success"]
        assert "error" in result
        assert "Unknown tool" in result["error"]
    
    @pytest.mark.asyncio
    async def test_tool_call_request(self, server):
        """Test tools/call protocol method."""
        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "explain_concept",
                "arguments": {"concept": "RNA-seq"}
            }
        }
        
        # Mock the handler to avoid actual API calls
        with patch.object(server, 'call_tool', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "success": True,
                "content": "RNA-seq is a technique..."
            }
            
            response = await server._handle_request(request)
            
            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 5
            assert "result" in response
            assert "content" in response["result"]


class TestMCPResourceReads:
    """Tests for MCP resource read handling."""
    
    @pytest.fixture
    def server(self):
        """Create server fixture."""
        return create_server()
    
    @pytest.mark.asyncio
    async def test_read_unknown_resource(self, server):
        """Test reading an unknown resource."""
        result = await server.read_resource("unknown://resource")
        
        assert "Unknown resource" in result
    
    @pytest.mark.asyncio
    async def test_resource_read_request(self, server):
        """Test resources/read protocol method."""
        request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "resources/read",
            "params": {
                "uri": "biopipelines://databases"
            }
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 6
        assert "result" in response
        assert "contents" in response["result"]
        
        # Database resource should return markdown content
        content = response["result"]["contents"][0]
        assert content["uri"] == "biopipelines://databases"
        assert "UniProt" in content["text"]


class TestMCPFormatters:
    """Tests for result formatting functions."""
    
    @pytest.fixture
    def server(self):
        """Create server fixture."""
        return create_server()
    
    def test_format_search_results_success(self, server):
        """Test search results formatting with success."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.count = 2
        mock_result.data = [
            {"id": "EXP001", "title": "Test experiment"},
            {"id": "EXP002", "name": "Another experiment"}
        ]
        
        formatted = server._format_search_results(mock_result)
        
        assert "Found 2 results" in formatted
        assert "EXP001" in formatted
        assert "EXP002" in formatted
    
    def test_format_search_results_failure(self, server):
        """Test search results formatting with failure."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.message = "API error"
        
        formatted = server._format_search_results(mock_result)
        
        assert "Search failed" in formatted
        assert "API error" in formatted
    
    def test_format_protein_results_success(self, server):
        """Test protein results formatting."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.count = 1
        mock_result.data = [
            {
                "primaryAccession": "P12345",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Test Protein"}
                    }
                },
                "genes": [{"geneName": {"value": "TESTP"}}]
            }
        ]
        
        formatted = server._format_protein_results(mock_result)
        
        assert "Found 1 proteins" in formatted
        assert "P12345" in formatted
        assert "TESTP" in formatted
    
    def test_format_enrichment_results(self, server):
        """Test enrichment results formatting."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.count = 3
        mock_result.data = [
            {"category": "GO:BP", "description": "cell cycle", "p_value": 0.001},
            {"category": "GO:BP", "description": "apoptosis", "p_value": 0.01},
            {"category": "KEGG", "description": "cancer pathway", "p_value": 0.05}
        ]
        
        formatted = server._format_enrichment_results(mock_result)
        
        assert "Found 3 enriched terms" in formatted
        assert "GO:BP" in formatted
        assert "cell cycle" in formatted
    
    def test_format_pathway_results(self, server):
        """Test KEGG pathway results formatting."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.count = 2
        mock_result.data = [
            {"id": "hsa04110", "name": "Cell cycle"},
            {"id": "hsa04210", "name": "Apoptosis"}
        ]
        
        formatted = server._format_pathway_results(mock_result)
        
        assert "Found 2 pathways" in formatted
        assert "hsa04110" in formatted
        assert "Cell cycle" in formatted


class TestMCPServerUnit:
    """Unit tests for server methods."""
    
    def test_register_tool(self):
        """Test tool registration."""
        server = create_server()
        initial_count = len(server.tools)
        
        async def dummy_handler(**kwargs):
            return {"success": True}
        
        server._register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            handler=dummy_handler
        )
        
        assert len(server.tools) == initial_count + 1
        assert "test_tool" in server.tools
    
    def test_register_resource(self):
        """Test resource registration."""
        server = create_server()
        initial_count = len(server.resources)
        
        async def dummy_handler():
            return "test content"
        
        server._register_resource(
            uri="test://resource",
            name="Test Resource",
            description="A test resource",
            handler=dummy_handler
        )
        
        assert len(server.resources) == initial_count + 1
        assert "test://resource" in server.resources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
