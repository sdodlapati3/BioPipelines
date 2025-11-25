"""
LLM Adapter Tests
=================

Unit tests for LLM adapters (OpenAI, vLLM, HuggingFace).

Run with:
    pytest tests/unit/test_llm_adapters.py -v
    
For integration tests (requires API keys / running servers):
    pytest tests/unit/test_llm_adapters.py -v --run-integration
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from workflow_composer.llm import (
    LLMAdapter,
    Message,
    LLMResponse,
    get_llm,
    list_providers,
    check_providers,
    OpenAIAdapter,
    VLLMAdapter,
    HuggingFaceAdapter,
)
from workflow_composer.llm.base import Role, MockLLMAdapter


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


@pytest.fixture
def mock_vllm_response():
    """Mock vLLM API response (OpenAI-compatible)."""
    return {
        "id": "cmpl-test",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from vLLM."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40
        }
    }


# ============================================================================
# Message and Response Tests
# ============================================================================

class TestMessage:
    """Test Message dataclass."""
    
    def test_create_system_message(self):
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are a helpful assistant."
    
    def test_create_user_message(self):
        msg = Message.user("Hello!")
        assert msg.role == Role.USER
        assert msg.content == "Hello!"
    
    def test_create_assistant_message(self):
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"
    
    def test_to_dict(self):
        msg = Message.user("Test")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Test"}


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_create_response(self):
        response = LLMResponse(
            content="Hello",
            model="test-model",
            provider="test",
            tokens_used=10
        )
        assert response.content == "Hello"
        assert response.text == "Hello"  # Alias
        assert response.model == "test-model"
        assert response.tokens_used == 10


# ============================================================================
# Factory Tests
# ============================================================================

class TestLLMFactory:
    """Test LLM factory functions."""
    
    def test_list_providers(self):
        providers = list_providers()
        assert "openai" in providers
        assert "vllm" in providers
        assert "huggingface" in providers
        assert "ollama" in providers
        assert "anthropic" in providers
    
    def test_get_llm_openai(self):
        # Should create adapter without error (even without API key)
        adapter = get_llm("openai", model="gpt-4o")
        assert adapter.provider_name == "openai"
        assert adapter.model == "gpt-4o"
    
    def test_get_llm_vllm(self):
        adapter = get_llm("vllm", model="meta-llama/Llama-3.1-8B-Instruct")
        assert adapter.provider_name == "vllm"
        assert adapter.model == "meta-llama/Llama-3.1-8B-Instruct"
    
    def test_get_llm_huggingface(self):
        adapter = get_llm("huggingface", model="mistralai/Mistral-7B-Instruct-v0.3")
        assert adapter.provider_name == "huggingface"
        assert adapter.model == "mistralai/Mistral-7B-Instruct-v0.3"
    
    def test_get_llm_default_models(self):
        # Each provider should have a default model
        for provider in ["openai", "vllm", "huggingface", "ollama", "anthropic"]:
            adapter = get_llm(provider)
            assert adapter.model is not None
            assert adapter.model != ""
    
    def test_get_llm_invalid_provider(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm("invalid_provider")


# ============================================================================
# Mock LLM Adapter Tests
# ============================================================================

class TestMockLLMAdapter:
    """Test MockLLMAdapter for testing."""
    
    def test_mock_complete(self):
        mock = MockLLMAdapter(responses={"hello": "Hi there!"})
        response = mock.complete("hello world")
        assert response.content == "Hi there!"
        assert "hello world" in mock.call_history
    
    def test_mock_default_response(self):
        mock = MockLLMAdapter()
        response = mock.complete("test prompt")
        assert response.content == "Mock response for testing"
    
    def test_mock_chat(self):
        mock = MockLLMAdapter(responses={"rna": "RNA-seq response"})
        messages = [Message.user("Tell me about rna-seq")]
        response = mock.chat(messages)
        assert response.content == "RNA-seq response"


# ============================================================================
# OpenAI Adapter Tests
# ============================================================================

class TestOpenAIAdapter:
    """Test OpenAI adapter."""
    
    def test_init_default(self):
        adapter = OpenAIAdapter()
        assert adapter.provider_name == "openai"
        assert adapter.model == "gpt-4o"
        assert adapter.temperature == 0.1
    
    def test_init_custom(self):
        adapter = OpenAIAdapter(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
            api_key="test-key"
        )
        assert adapter.model == "gpt-4o"
        assert adapter.temperature == 0.5
        assert adapter.max_tokens == 2048
        assert adapter.api_key == "test-key"
    
    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            adapter = OpenAIAdapter()
            assert adapter.api_key == "env-key"
    
    @patch('urllib.request.urlopen')
    def test_chat_success(self, mock_urlopen, mock_openai_response):
        # Setup mock
        mock_response = Mock()
        mock_response.read.return_value = __import__('json').dumps(mock_openai_response).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        adapter = OpenAIAdapter(api_key="test-key")
        messages = [Message.user("Hello")]
        response = adapter.chat(messages)
        
        assert response.content == "This is a test response."
        assert response.model == "gpt-4o"
        assert response.tokens_used == 30
    
    def test_chat_no_api_key(self):
        adapter = OpenAIAdapter(api_key=None)
        adapter.api_key = None  # Ensure no key
        
        with pytest.raises(ValueError, match="API key not configured"):
            adapter.chat([Message.user("test")])


# ============================================================================
# vLLM Adapter Tests
# ============================================================================

class TestVLLMAdapter:
    """Test vLLM adapter."""
    
    def test_init_default(self):
        adapter = VLLMAdapter()
        assert adapter.provider_name == "vllm"
        assert adapter.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert adapter.base_url == "http://localhost:8000"
    
    def test_init_custom(self):
        adapter = VLLMAdapter(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            base_url="http://gpu-node:8000",
            temperature=0.5,
            max_tokens=2048
        )
        assert adapter.model == "mistralai/Mistral-7B-Instruct-v0.3"
        assert adapter.base_url == "http://gpu-node:8000"
    
    def test_model_alias(self):
        adapter = VLLMAdapter(model="llama3.1-8b")
        assert adapter.model == "meta-llama/Llama-3.1-8B-Instruct"
    
    def test_recommended_models(self):
        models = VLLMAdapter.get_recommended_models()
        assert "llama3.1-8b" in models
        assert "mistral-7b" in models
        assert "codellama-34b" in models
    
    def test_launch_command(self):
        cmd = VLLMAdapter.get_launch_command(
            model="meta-llama/Llama-3.1-8B-Instruct",
            port=8000,
            tensor_parallel_size=2
        )
        assert "vllm.entrypoints.openai.api_server" in cmd
        assert "--model meta-llama/Llama-3.1-8B-Instruct" in cmd
        assert "--tensor-parallel-size 2" in cmd
    
    @patch('urllib.request.urlopen')
    def test_chat_success(self, mock_urlopen, mock_vllm_response):
        # Setup mock
        mock_response = Mock()
        mock_response.read.return_value = __import__('json').dumps(mock_vllm_response).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        adapter = VLLMAdapter()
        messages = [Message.user("Hello")]
        response = adapter.chat(messages)
        
        assert response.content == "This is a test response from vLLM."
        assert "Llama-3.1-8B" in response.model
        assert response.tokens_used == 40


# ============================================================================
# HuggingFace Adapter Tests
# ============================================================================

class TestHuggingFaceAdapter:
    """Test HuggingFace adapter with vLLM backend support."""
    
    def test_init_default(self):
        adapter = HuggingFaceAdapter()
        assert adapter.provider_name == "huggingface"
        assert "Llama-3" in adapter.model
    
    def test_init_vllm_backend(self):
        adapter = HuggingFaceAdapter(
            model="meta-llama/Llama-3.1-8B-Instruct",
            backend="vllm",
            vllm_url="http://localhost:8000"
        )
        assert adapter.backend == "vllm"
        assert adapter.vllm_url == "http://localhost:8000"
    
    def test_deprecated_use_api(self):
        # Test backward compatibility
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = HuggingFaceAdapter(use_api=True)
            assert adapter.backend == "api"
    
    def test_backend_options(self):
        # API backend
        adapter = HuggingFaceAdapter(backend="api")
        assert adapter.backend == "api"
        
        # Transformers backend
        adapter = HuggingFaceAdapter(backend="transformers")
        assert adapter.backend == "transformers"
        
        # vLLM backend
        adapter = HuggingFaceAdapter(backend="vllm")
        assert adapter.backend == "vllm"


# ============================================================================
# Integration Tests (require API keys / running servers)
# ============================================================================

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require API keys or running servers"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests for OpenAI (requires OPENAI_API_KEY)."""
    
    def test_complete(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        
        adapter = OpenAIAdapter(model="gpt-4o")
        response = adapter.complete("Say 'test' and nothing else.")
        assert "test" in response.content.lower()
    
    def test_chat(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        
        adapter = OpenAIAdapter(model="gpt-4o")
        messages = [
            Message.system("You are a helpful assistant. Be brief."),
            Message.user("What is RNA-seq? One sentence only.")
        ]
        response = adapter.chat(messages)
        assert len(response.content) > 10
        assert response.tokens_used > 0


@pytest.mark.integration
class TestVLLMIntegration:
    """Integration tests for vLLM (requires running vLLM server)."""
    
    def test_is_available(self):
        adapter = VLLMAdapter()
        # This will return False if server isn't running, which is OK
        available = adapter.is_available()
        if not available:
            pytest.skip("vLLM server not running")
    
    def test_complete(self):
        adapter = VLLMAdapter()
        if not adapter.is_available():
            pytest.skip("vLLM server not running")
        
        response = adapter.complete("Say 'hello' and nothing else.")
        assert "hello" in response.content.lower()
    
    def test_get_models(self):
        adapter = VLLMAdapter()
        if not adapter.is_available():
            pytest.skip("vLLM server not running")
        
        models = adapter.get_models()
        assert len(models) > 0


# ============================================================================
# Bioinformatics-Specific Tests
# ============================================================================

class TestBioinformaticsPrompts:
    """Test LLM responses for bioinformatics-specific prompts."""
    
    def test_mock_rnaseq_workflow(self):
        mock = MockLLMAdapter(responses={
            "rna-seq": "For RNA-seq analysis, use: 1) FastQC 2) STAR 3) featureCounts 4) DESeq2"
        })
        response = mock.complete("Create an RNA-seq workflow")
        assert "RNA-seq" in response.content or "rna-seq" in response.content.lower()
    
    def test_mock_tool_suggestion(self):
        mock = MockLLMAdapter(responses={
            "variant": "For variant calling, I recommend GATK HaplotypeCaller or DeepVariant."
        })
        response = mock.complete("What tool for variant calling?")
        assert "variant" in response.content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
