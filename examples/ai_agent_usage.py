"""
Example: AI Agent using BioPipelines containers for multi-omics analysis

This demonstrates how an AI agent can:
1. Discover available pipelines
2. Compose multi-omics workflows
3. Execute and monitor analysis
4. Interpret results
"""

from pathlib import Path
from biopipelines.containers import ContainerRegistry


def example_1_simple_discovery():
    """Example 1: AI agent discovers RNA-seq pipeline"""
    print("=== Example 1: Simple Pipeline Discovery ===\n")
    
    # Initialize registry
    registry = ContainerRegistry(Path("containers/"))
    
    # User query: "I have RNA-seq data"
    user_query = "I want to analyze RNA-seq data to find differentially expressed genes"
    
    # AI agent discovers appropriate container
    recommendations = registry.recommend_for_query(user_query)
    
    for container in recommendations:
        print(f"Recommended: {container.name}")
        print(f"  Category: {container.category}")
        print(f"  Capabilities: {', '.join(container.capabilities)}")
        print(f"  Tools: {', '.join([t.name for t in container.tools])}")
        print()


def example_2_capability_search():
    """Example 2: Search by specific capability"""
    print("=== Example 2: Capability-Based Search ===\n")
    
    registry = ContainerRegistry(Path("containers/"))
    
    # Find all containers that can do alignment
    alignment_containers = registry.search(capability="alignment")
    
    print(f"Found {len(alignment_containers)} containers with alignment capability:")
    for container in alignment_containers:
        print(f"  - {container.name}")
        tools = [t.name for t in container.tools if t.stage == "alignment"]
        print(f"    Alignment tools: {', '.join(tools)}")
    print()


def example_3_execute_pipeline():
    """Example 3: AI agent executes a pipeline"""
    print("=== Example 3: Pipeline Execution ===\n")
    
    registry = ContainerRegistry(Path("containers/"))
    
    # Get RNA-seq container
    rna_seq = registry.get_container("biopipelines-rna-seq")
    
    # AI agent prepares execution parameters
    params = {
        "input": "/data/fastq",
        "output": "/data/results",
        "genome": "hg38",
        "threads": 16,
        "strandedness": "reverse"
    }
    
    # Generate execution command
    command = rna_seq.get_execution_command(params)
    
    print("AI Agent will execute:")
    print(f"  {command}\n")
    
    # Check resource requirements
    resources = rna_seq.get_resource_requirements()
    print("Resource requirements:")
    print(f"  Memory: {resources['recommended_memory_gb']} GB")
    print(f"  Cores: {resources['recommended_cores']}")
    print(f"  Disk: {resources['disk_space_gb']} GB")
    print(f"  Estimated time: {resources['estimated_runtime_hours']} hours")
    print()


def example_4_multi_omics_workflow():
    """Example 4: AI agent composes multi-omics workflow"""
    print("=== Example 4: Multi-Omics Workflow Composition ===\n")
    
    registry = ContainerRegistry(Path("containers/"))
    
    # User request: "I have RNA-seq and ChIP-seq data, find TF-target relationships"
    
    # Step 1: AI agent identifies required containers
    rna_seq = registry.search(category="transcriptomics")[0]
    chip_seq = registry.search(capability="peak_calling")[0]  # When implemented
    
    print("AI Agent composes workflow:")
    print(f"  Step 1: {rna_seq.name} → Identify DE genes")
    print(f"  Step 2: {chip_seq.name} → Find TF binding sites")  
    print(f"  Step 3: Integration → Map TF-target relationships")
    print()
    
    # Step 2: Generate execution plan
    workflow = {
        "name": "TF-target discovery",
        "steps": [
            {
                "container": rna_seq.name,
                "params": {
                    "input": "/data/rna_fastq",
                    "output": "/data/rna_results",
                    "genome": "hg38"
                },
                "output_key": "de_genes"
            },
            {
                "container": "biopipelines-chip-seq",  # When implemented
                "params": {
                    "input": "/data/chip_fastq",
                    "output": "/data/chip_results",
                    "genome": "hg38"
                },
                "output_key": "tf_peaks"
            }
        ]
    }
    
    print("Workflow plan generated:")
    for i, step in enumerate(workflow["steps"], 1):
        print(f"  {i}. Execute {step['container']}")
        print(f"     Output: {step['output_key']}")
    print()


def example_5_ai_interpretation():
    """Example 5: AI agent interprets results"""
    print("=== Example 5: Results Interpretation ===\n")
    
    registry = ContainerRegistry(Path("containers/"))
    rna_seq = registry.get_container("biopipelines-rna-seq")
    
    # Get AI hints about outputs
    hints = rna_seq.get_ai_hints()
    
    print("AI Agent reads container hints:")
    print(f"  Use case: {hints['typical_use_case']}")
    print(f"\n  Key outputs for interpretation:")
    for output in hints['key_outputs']:
        print(f"    - {output}")
    
    print(f"\n  Common issues to watch for:")
    for issue in hints['common_issues']:
        print(f"    ⚠ {issue}")
    
    print(f"\n  Can be combined with:")
    for integration in hints['composition_with']:
        print(f"    + {integration}")
    print()


def example_6_container_metadata():
    """Example 6: Explore container metadata"""
    print("=== Example 6: Container Metadata Exploration ===\n")
    
    registry = ContainerRegistry(Path("containers/"))
    
    print(f"Registry contains {len(registry)} containers\n")
    
    print("Available categories:")
    for category in registry.get_categories():
        containers = registry.search(category=category)
        print(f"  - {category}: {len(containers)} containers")
    
    print("\nAll capabilities across containers:")
    for capability in registry.get_capabilities()[:10]:  # Show first 10
        print(f"  - {capability}")
    
    print("\nContainers with samtools:")
    samtools_containers = registry.find_by_tool("samtools")
    for container in samtools_containers:
        print(f"  - {container.name}")
    print()


def example_7_programmatic_workflow():
    """Example 7: Programmatic workflow execution"""
    print("=== Example 7: Programmatic Workflow (Pseudocode) ===\n")
    
    code_example = '''
from biopipelines.containers import ContainerRegistry
from biopipelines.agents import PipelineOrchestrator

# Initialize
registry = ContainerRegistry("containers/")
orchestrator = PipelineOrchestrator(registry)

# AI agent receives user request
user_request = """
I have paired-end RNA-seq data from 6 samples (3 treated, 3 control).
I want to find differentially expressed genes with FDR < 0.05.
"""

# Agent discovers and validates
pipeline = orchestrator.discover_pipeline(user_request)
print(f"Using: {pipeline.name}")

# Agent creates execution plan
plan = orchestrator.create_plan(
    pipeline=pipeline,
    inputs=["sample1_R1.fastq.gz", "sample1_R2.fastq.gz", ...],
    conditions={"treated": 3, "control": 3},
    genome="hg38"
)

# Execute with monitoring
result = orchestrator.execute(plan, monitor=True)

# Interpret results
if result.success:
    summary = f"""
    Analysis complete:
    - {result.metrics['total_reads']} million reads processed
    - {result.metrics['mapping_rate']}% alignment rate
    - {result.metrics['de_genes']} DE genes found (FDR < 0.05)
    - {result.metrics['upregulated']} upregulated
    - {result.metrics['downregulated']} downregulated
    
    Top results saved to: {result.output_dir}/deseq2/
    """
    print(summary)
    '''
    
    print(code_example)


if __name__ == "__main__":
    # Run all examples
    examples = [
        example_1_simple_discovery,
        example_2_capability_search,
        example_3_execute_pipeline,
        example_4_multi_omics_workflow,
        example_5_ai_interpretation,
        example_6_container_metadata,
        example_7_programmatic_workflow
    ]
    
    for example in examples:
        try:
            example()
            print("-" * 60)
            print()
        except Exception as e:
            print(f"Example failed: {e}\n")
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
