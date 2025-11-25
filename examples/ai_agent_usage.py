#!/usr/bin/env python3
"""
Example: AI-Powered Workflow Generation with BioPipelines

This demonstrates how to use the Workflow Composer to:
1. Generate workflows from natural language descriptions
2. Use different LLM providers (OpenAI, vLLM)
3. Parse analysis intents
4. Search tools and modules
"""

import os
from pathlib import Path

# Note: Run from BioPipelines root directory
# or install: pip install -e .


def example_1_basic_workflow_generation():
    """Example 1: Generate a basic RNA-seq workflow."""
    print("=== Example 1: Basic Workflow Generation ===\n")
    
    from workflow_composer import Composer
    from workflow_composer.llm import get_llm, check_providers
    
    # Check available providers
    available = check_providers()
    print(f"Available LLM providers: {available}")
    
    # Select available provider
    if available.get("openai"):
        llm = get_llm("openai", model="gpt-4o")
        print("Using: OpenAI GPT-4o")
    elif available.get("vllm"):
        llm = get_llm("vllm", model="llama3.1-8b")
        print("Using: vLLM Llama 3.1 8B")
    else:
        print("No LLM provider available. Set OPENAI_API_KEY or start vLLM server.")
        return
    
    # Create composer
    composer = Composer(llm=llm)
    
    # Generate workflow from natural language
    description = """
    RNA-seq differential expression analysis:
    - Organism: Mouse
    - Data: Paired-end Illumina reads
    - Comparison: Treatment vs Control
    - Tools: STAR, featureCounts, DESeq2
    """
    
    print(f"\nGenerating workflow for:\n{description}")
    
    # Check readiness
    readiness = composer.check_readiness(description)
    print(f"\nReadiness check:")
    print(f"  Ready: {readiness['ready']}")
    print(f"  Tools found: {readiness.get('tools_found', 0)}")
    print(f"  Modules found: {readiness.get('modules_found', 0)}")
    
    # Generate and save (uncomment to actually generate)
    # workflow = composer.generate(description, output_dir="examples/generated/rnaseq_demo/")
    # print(f"\nWorkflow saved to: {workflow.output_dir}")


def example_2_openai_usage():
    """Example 2: Using OpenAI specifically."""
    print("\n=== Example 2: OpenAI Provider ===\n")
    
    from workflow_composer.llm import get_llm, OpenAIAdapter, Message
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Skipping OpenAI example.")
        return
    
    # Create OpenAI adapter
    llm = get_llm("openai", model="gpt-4o")
    print(f"Adapter: {llm}")
    
    # Simple completion
    response = llm.complete(
        "List 3 tools commonly used for RNA-seq alignment. Just names, one per line."
    )
    print(f"\nRNA-seq alignment tools:\n{response.content}")
    
    # Chat with context
    messages = [
        Message.system("You are a bioinformatics expert. Be concise."),
        Message.user("What's the difference between STAR and Bowtie2?")
    ]
    response = llm.chat(messages)
    print(f"\nSTAR vs Bowtie2:\n{response.content[:300]}...")


def example_3_vllm_usage():
    """Example 3: Using vLLM with open-source models."""
    print("\n=== Example 3: vLLM Provider ===\n")
    
    from workflow_composer.llm import VLLMAdapter
    
    # Show recommended models
    models = VLLMAdapter.get_recommended_models()
    print("Recommended models for vLLM:")
    for alias, full_name in models.items():
        print(f"  {alias}: {full_name}")
    
    # Generate launch command
    cmd = VLLMAdapter.get_launch_command(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )
    print(f"\nvLLM launch command:\n  {cmd}")
    
    # Check if vLLM server is running
    try:
        llm = VLLMAdapter(model="llama3.1-8b")
        if llm.is_available():
            print("\nvLLM server is running!")
            response = llm.complete("What is the capital of France?")
            print(f"Test response: {response.content}")
        else:
            print("\nvLLM server not available. Start with:")
            print(f"  {cmd}")
    except Exception as e:
        print(f"\nvLLM not available: {e}")


def example_4_intent_parsing():
    """Example 4: Parse natural language into structured intent."""
    print("\n=== Example 4: Intent Parsing ===\n")
    
    from workflow_composer import Composer
    from workflow_composer.llm import get_llm, check_providers
    
    available = check_providers()
    if not any(available.values()):
        print("No LLM provider available.")
        return
    
    # Use first available provider
    provider = next(k for k, v in available.items() if v)
    llm = get_llm(provider)
    composer = Composer(llm=llm)
    
    # Parse intent
    description = "ChIP-seq peak calling for human H3K4me3 samples with input control"
    
    intent = composer.parse_intent(description)
    print(f"Parsed intent for: '{description}'")
    print(f"  Analysis type: {intent.analysis_type.value}")
    print(f"  Organism: {intent.organism}")
    print(f"  Genome build: {intent.genome_build}")
    print(f"  Data type: {intent.data_type}")
    print(f"  Has comparison: {intent.has_comparison}")
    print(f"  Confidence: {intent.confidence:.2f}")


def example_5_tool_search():
    """Example 5: Search for tools in the catalog."""
    print("\n=== Example 5: Tool Search ===\n")
    
    from workflow_composer import Composer
    
    # Create composer (doesn't need LLM for tool search)
    try:
        composer = Composer()
    except Exception:
        print("Composer initialization requires valid config. Using mock data.")
        return
    
    # Search for alignment tools
    print("Searching for 'alignment' tools...")
    matches = composer.tool_selector.fuzzy_search("alignment", limit=10)
    
    for match in matches[:5]:
        print(f"  {match.tool.name:20} ({match.tool.container:15}) score: {match.score:.2f}")
    
    # Find tools for specific analysis
    print("\nTools for RNA-seq analysis:")
    tool_map = composer.find_tools("rnaseq")
    for category, tools in tool_map.items():
        tool_names = [t.name for t in tools[:3]]
        print(f"  {category}: {', '.join(tool_names)}")


def example_6_module_listing():
    """Example 6: List available Nextflow modules."""
    print("\n=== Example 6: Module Listing ===\n")
    
    from workflow_composer import Composer
    
    try:
        composer = Composer()
    except Exception:
        print("Composer initialization requires valid config.")
        return
    
    # List modules by category
    modules = composer.module_mapper.list_by_category()
    
    print("Available modules:")
    for category, mods in sorted(modules.items()):
        print(f"  {category}: {len(mods)} modules")
        for mod in mods[:3]:
            print(f"    - {mod}")
        if len(mods) > 3:
            print(f"    ... and {len(mods) - 3} more")


def example_7_switching_providers():
    """Example 7: Switching between LLM providers."""
    print("\n=== Example 7: Provider Switching ===\n")
    
    from workflow_composer import Composer
    from workflow_composer.llm import get_llm, check_providers
    
    available = check_providers()
    print(f"Available providers: {list(k for k, v in available.items() if v)}")
    
    if not any(available.values()):
        print("No providers available.")
        return
    
    # Start with one provider
    provider = next(k for k, v in available.items() if v)
    composer = Composer(llm=get_llm(provider))
    print(f"\nStarting with: {composer.llm}")
    
    # Switch to another if available
    other_providers = [k for k, v in available.items() if v and k != provider]
    if other_providers:
        new_provider = other_providers[0]
        composer.switch_llm(new_provider)
        print(f"Switched to: {composer.llm}")


def example_8_complete_workflow():
    """Example 8: Complete workflow generation example."""
    print("\n=== Example 8: Complete Workflow Generation ===\n")
    
    from workflow_composer import Composer
    from workflow_composer.llm import get_llm, check_providers
    
    available = check_providers()
    if not available.get("openai") and not available.get("vllm"):
        print("Need OpenAI or vLLM for workflow generation.")
        return
    
    # Choose best available provider
    if available.get("openai"):
        llm = get_llm("openai", model="gpt-4o")
    else:
        llm = get_llm("vllm", model="llama3.1-8b")
    
    composer = Composer(llm=llm)
    
    # Define analysis
    description = """
    ATAC-seq chromatin accessibility analysis:
    - Organism: Human (GRCh38)
    - Paired-end data
    - Steps:
      1. FastQC quality control
      2. Trim adapters with fastp
      3. Align with Bowtie2
      4. Remove duplicates
      5. Call peaks with MACS2
      6. Generate signal tracks
    """
    
    print(f"Generating workflow for:\n{description}")
    
    # Check readiness
    readiness = composer.check_readiness(description)
    print(f"\nReadiness: {readiness['ready']}")
    
    if not readiness['ready']:
        print(f"Issues: {readiness['issues']}")
        return
    
    # Generate workflow
    output_dir = "examples/generated/atacseq_demo/"
    workflow = composer.generate(description, output_dir=output_dir)
    
    print(f"\n✓ Workflow generated!")
    print(f"  Name: {workflow.name}")
    print(f"  Modules: {len(workflow.modules_used)}")
    print(f"  Output: {workflow.output_dir}")
    
    # Show generated files
    if workflow.output_dir and workflow.output_dir.exists():
        print(f"\nGenerated files:")
        for f in workflow.output_dir.iterdir():
            print(f"  {f.name}")


if __name__ == "__main__":
    # Run all examples
    examples = [
        example_1_basic_workflow_generation,
        example_2_openai_usage,
        example_3_vllm_usage,
        example_4_intent_parsing,
        example_5_tool_search,
        example_6_module_listing,
        example_7_switching_providers,
        # example_8_complete_workflow,  # Uncomment to generate actual workflow
    ]
    
    for example in examples:
        try:
            example()
            print("-" * 60)
        except Exception as e:
            print(f"Example failed: {e}\n")
            print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
