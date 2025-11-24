# Dynamic Container Generation Strategy for AI-Driven Pipelines

**Date**: November 24, 2025  
**Context**: Extending Nextflow architecture plan with runtime container customization  
**Purpose**: Enable AI agents to create custom containers for user-specific pipeline requests

---

## 1. The Problem Statement

### Current Situation
- **Fixed containers**: 12 pre-built SIF files (rna-seq, dna-seq, chip-seq, etc.)
- **Monolithic design**: Each container has ALL tools for that pipeline type (10-18 GB each)
- **Limited flexibility**: What if user requests:
  - "Run RNA-seq but use Salmon instead of STAR"
  - "I need STAR + Salmon + Kallisto for tool comparison"
  - "RNA-seq + my custom R script for novel normalization"
  - "ChIP-seq with new tool XYZ from GitHub"

### The Challenge
When AI agent generates a **custom workflow**, it may need:
1. **Tool substitution**: Different aligner than standard pipeline
2. **Tool addition**: Extra tools not in base container
3. **Custom scripts**: User-provided code
4. **Novel combinations**: Tools from multiple existing containers
5. **Latest versions**: Bleeding-edge tool releases

**Question**: How do we provide containers for these custom workflows **dynamically**?

---

## 2. Proposed Multi-Tier Container Architecture

### Tier 1: Base Foundation (Pre-built, Read-only)

**Purpose**: Shared foundation with universal tools

```
containers/base/base.def → base_1.0.0.sif (2 GB)

Contents:
- samtools, bcftools, bedtools, htslib
- fastqc, multiqc
- Python 3.11 + NumPy/Pandas/Biopython
- R 4.3 + ggplot2/pheatmap
- Common aligners: bowtie2, bwa
- Preprocessing: fastp, trimmomatic, cutadapt
```

**Usage**: ALL containers inherit from this (FROM base_1.0.0.sif)

**Lifecycle**: Updated quarterly, versioned, immutable

---

### Tier 2: Domain Modules (Pre-built, Read-only)

**Purpose**: Specialized tool suites for common analysis types

```
containers/modules/
├── alignment_short_read_1.0.0.sif     # 4 GB
│   ├── Inherits: base_1.0.0.sif
│   ├── Adds: STAR, HISAT2, Salmon, Kallisto, Subread
│
├── alignment_long_read_1.0.0.sif      # 3 GB
│   ├── Inherits: base_1.0.0.sif
│   ├── Adds: Minimap2, Winnowmap, GraphMap2
│
├── variant_calling_1.0.0.sif          # 5 GB
│   ├── Inherits: base_1.0.0.sif
│   ├── Adds: GATK 4.5, FreeBayes, VEP, SnpEff
│
├── peak_calling_1.0.0.sif             # 3 GB
│   ├── Inherits: base_1.0.0.sif
│   ├── Adds: MACS2, MACS3, HOMER, SICER, GEM
│
├── assembly_1.0.0.sif                 # 4 GB
│   ├── Inherits: base_1.0.0.sif
│   ├── Adds: SPAdes, Flye, Canu, Unicycler
│
├── scrna_analysis_1.0.0.sif           # 8 GB
│   ├── Inherits: base_1.0.0.sif
│   ├── Adds: CellRanger, Scanpy, Seurat, Monocle3
│
└── [more domain modules]
```

**Characteristics**:
- Pre-built and cached on shared storage
- Cover 90% of common use cases
- AI agent selects appropriate modules for workflow
- Multiple modules can be used in same workflow (different processes)

**Lifecycle**: Updated monthly, versioned, immutable

---

### Tier 3: Dynamic Custom Containers (Built on-demand, Ephemeral)

**Purpose**: User-specific customizations for novel workflows

#### 3A. Overlay Approach (Fast, Recommended)

**Strategy**: Use OverlayFS to add tools on top of existing container **without rebuilding**

```bash
# AI agent creates overlay with additional tools
singularity run --overlay overlay.img:rw base_1.0.0.sif \
    micromamba install -y new_tool_xyz

# Overlay contains only the delta (new tools)
# Base container remains unchanged
# Total space: Base (2 GB) + Overlay (500 MB) = 2.5 GB
```

**Advantages**:
- ✅ **Fast**: Minutes to add tools, not hours to rebuild
- ✅ **Efficient**: Only store delta, not entire container
- ✅ **Flexible**: Multiple overlays can be created from same base
- ✅ **Cacheable**: Same overlay can be reused for similar requests

**Use Cases**:
- User wants specific tool version: "Use STAR 2.7.11b instead of 2.7.10a"
- Add custom script: User uploads `custom_normalize.R`
- Add new tool: "Include RSEM for isoform quantification"

#### 3B. Microservice Containers (Ultra-light, For specific tools)

**Strategy**: Tiny containers (100-500 MB) with single tool or script

```
containers/microservices/
├── star_2.7.11b.sif              # 450 MB - STAR only
├── rsem_1.3.3.sif                # 300 MB - RSEM only
├── custom_user_script_12345.sif   # 150 MB - User's R script + deps
└── github_tool_abc.sif           # 200 MB - Novel tool from GitHub
```

**Nextflow Integration**:
```groovy
process STANDARD_STAR {
    container "alignment_short_read_1.0.0.sif"  // Use domain module
    // ...
}

process SPECIFIC_STAR_VERSION {
    container "star_2.7.11b.sif"  // Use microservice for specific version
    // ...
}

process USER_CUSTOM_ANALYSIS {
    container "custom_user_script_12345.sif"  // User's custom code
    // ...
}
```

**Advantages**:
- ✅ **Minimal size**: Only what's needed (100-500 MB vs 2-5 GB)
- ✅ **Fast build**: 2-5 minutes to create
- ✅ **Version-specific**: Pin exact tool version per-process
- ✅ **Isolated**: Dependency conflicts can't affect other tools

#### 3C. Just-In-Time (JIT) Container Build

**Strategy**: AI agent builds container **before workflow execution**

**Workflow**:
```
1. User Query: "RNA-seq with Salmon + my custom normalization script"
   
2. AI Planner Agent:
   - Identifies required tools: FastQC, Salmon, MultiQC, custom R script
   - Checks cache: Do we have these exact tools?
   
3. AI Container Builder Agent:
   - Option A: Found cached → Use existing containers
   - Option B: Need new tool → Build microservice container
   - Option C: Need many tools → Build overlay on domain module
   
4. Container Build (parallel with workflow generation):
   - Start from: alignment_short_read_1.0.0.sif (has Salmon)
   - Create overlay: Add user's custom_normalize.R + dependencies
   - Time: 3-5 minutes
   - Output: custom_rnaseq_user123_v1.sif or overlay file
   
5. Nextflow Workflow Launch:
   - Workflow uses new container
   - Container cached for future identical requests
   
6. Post-execution:
   - Keep container for 30 days (cache)
   - If unused, garbage collect
```

**Build Time Optimization**:
```
Fast path (2-5 min):
- Base/module already exists → Create overlay with new tools
- OR: Use microservice containers (each 2-5 min to build)

Medium path (10-20 min):
- Build small custom container from base
- Add 5-10 tools via micromamba

Slow path (30-60 min):
- Full container rebuild (rare, only for complex requirements)
- Last resort: User waits or job queues until ready
```

---

## 3. AI Agent Integration

### Container Selection Logic

```python
class ContainerStrategyAgent:
    """AI agent that determines optimal container strategy for workflow"""
    
    def select_containers(self, workflow_design):
        """
        Given a workflow design, select or build containers for each process.
        
        Returns:
            container_plan: {
                'process_name': {
                    'strategy': 'existing' | 'overlay' | 'microservice' | 'jit_build',
                    'container': 'path/to/container.sif',
                    'build_time_estimate': int (seconds),
                    'cache_hit': bool
                }
            }
        """
        container_plan = {}
        
        for process in workflow_design.processes:
            tools_needed = process.tools
            
            # 1. Check if existing container has ALL tools
            existing = self.find_existing_container(tools_needed)
            if existing and existing.has_all_tools(tools_needed):
                container_plan[process.name] = {
                    'strategy': 'existing',
                    'container': existing.path,
                    'build_time_estimate': 0,  # Instant
                    'cache_hit': True
                }
                continue
            
            # 2. Check if we can overlay on existing
            base_container = self.find_closest_container(tools_needed)
            missing_tools = set(tools_needed) - set(base_container.tools)
            
            if len(missing_tools) <= 3 and all(self.is_easy_to_install(t) for t in missing_tools):
                # Overlay strategy
                overlay_path = self.check_overlay_cache(base_container, missing_tools)
                if overlay_path:
                    container_plan[process.name] = {
                        'strategy': 'overlay',
                        'container': overlay_path,
                        'build_time_estimate': 0,
                        'cache_hit': True
                    }
                else:
                    container_plan[process.name] = {
                        'strategy': 'overlay',
                        'container': f'{base_container.path}+overlay',
                        'build_time_estimate': 180,  # 3 minutes
                        'cache_hit': False,
                        'build_spec': {
                            'base': base_container.path,
                            'add_tools': list(missing_tools)
                        }
                    }
                continue
            
            # 3. Check if single tool → microservice
            if len(tools_needed) == 1:
                tool = tools_needed[0]
                microservice = self.check_microservice_cache(tool)
                if microservice:
                    container_plan[process.name] = {
                        'strategy': 'microservice',
                        'container': microservice,
                        'build_time_estimate': 0,
                        'cache_hit': True
                    }
                else:
                    container_plan[process.name] = {
                        'strategy': 'microservice',
                        'container': f'microservices/{tool.name}_{tool.version}.sif',
                        'build_time_estimate': 240,  # 4 minutes
                        'cache_hit': False,
                        'build_spec': {
                            'type': 'microservice',
                            'tool': tool
                        }
                    }
                continue
            
            # 4. Last resort: JIT build custom container
            container_plan[process.name] = {
                'strategy': 'jit_build',
                'container': f'custom/{workflow_design.id}_{process.name}.sif',
                'build_time_estimate': 900,  # 15 minutes
                'cache_hit': False,
                'build_spec': {
                    'type': 'custom',
                    'base': self.select_best_base(tools_needed),
                    'tools': tools_needed
                }
            }
        
        return container_plan
    
    def estimate_total_prep_time(self, container_plan):
        """Calculate total time needed before workflow can start"""
        # Builds can happen in parallel
        uncached_builds = [
            plan['build_time_estimate'] 
            for plan in container_plan.values() 
            if not plan['cache_hit']
        ]
        
        if not uncached_builds:
            return 0  # All cached, instant start
        
        # Builds run in parallel, return max
        return max(uncached_builds)
    
    def build_containers(self, container_plan):
        """Build all non-cached containers in parallel"""
        import concurrent.futures
        
        build_tasks = [
            (name, plan) 
            for name, plan in container_plan.items() 
            if not plan['cache_hit']
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._build_container, name, plan): name
                for name, plan in build_tasks
            }
            
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                result = future.result()
                print(f"Container for {name}: {result.status}")
```

### Container Builder Agent

```python
class ContainerBuilderAgent:
    """Builds custom containers based on specifications"""
    
    def build_overlay(self, base_container, tools_to_add):
        """
        Create overlay with additional tools on existing container.
        Fast: 2-5 minutes for most tools.
        """
        overlay_name = f"overlay_{hash(base_container + str(tools_to_add))}.img"
        overlay_path = f"/scratch/container_cache/overlays/{overlay_name}"
        
        # Create overlay image
        subprocess.run([
            "singularity", "overlay", "create",
            "--size", "2048",  # 2 GB overlay
            overlay_path
        ])
        
        # Install tools into overlay
        install_cmd = " && ".join([
            f"micromamba install -y -n base -c conda-forge -c bioconda {tool}"
            for tool in tools_to_add
        ])
        
        subprocess.run([
            "singularity", "exec",
            "--overlay", f"{overlay_path}:rw",
            base_container,
            "/bin/bash", "-c", install_cmd
        ])
        
        # Cache the overlay
        self.cache_overlay(overlay_path, base_container, tools_to_add)
        
        return overlay_path
    
    def build_microservice(self, tool_spec):
        """
        Build ultra-light container with single tool.
        Fast: 3-5 minutes for most tools.
        """
        def_content = f"""
Bootstrap: docker
From: mambaorg/micromamba:1.5.8

%post
    micromamba install -y -n base -c conda-forge -c bioconda \\
        {tool_spec.name}={tool_spec.version}
    micromamba clean --all --yes

%runscript
    exec {tool_spec.executable} "$@"
"""
        
        def_file = f"/tmp/{tool_spec.name}.def"
        sif_file = f"/scratch/container_cache/microservices/{tool_spec.name}_{tool_spec.version}.sif"
        
        Path(def_file).write_text(def_content)
        
        subprocess.run([
            "singularity", "build",
            "--fakeroot",  # Rootless build
            sif_file,
            def_file
        ])
        
        return sif_file
    
    def build_custom_container(self, base_container, tools, custom_scripts=None):
        """
        Build full custom container.
        Slower: 10-30 minutes depending on tools.
        """
        def_content = f"""
Bootstrap: localimage
From: {base_container}

%files
"""
        # Add custom scripts if provided
        if custom_scripts:
            for script in custom_scripts:
                def_content += f"    {script.local_path} {script.container_path}\n"
        
        def_content += f"""
%post
    # Install requested tools
    micromamba install -y -n base -c conda-forge -c bioconda \\
"""
        def_content += " \\\n".join([f"        {t.name}={t.version}" for t in tools])
        
        def_content += """
    
    # Install custom script dependencies
"""
        if custom_scripts:
            for script in custom_scripts:
                if script.dependencies:
                    def_content += f"    micromamba install -y -n base {' '.join(script.dependencies)}\n"
        
        def_content += """
    micromamba clean --all --yes

%environment
    export PATH="/opt/conda/bin:$PATH"
"""
        
        # Build container
        def_file = f"/tmp/custom_{uuid.uuid4()}.def"
        sif_file = f"/scratch/container_cache/custom/{uuid.uuid4()}.sif"
        
        Path(def_file).write_text(def_content)
        
        subprocess.run([
            "singularity", "build",
            "--fakeroot",
            sif_file,
            def_file
        ])
        
        return sif_file
```

---

## 4. Caching Strategy

### Cache Hierarchy

```
/scratch/container_cache/
├── base/                           # Tier 1 - Never deleted
│   └── base_1.0.0.sif
│
├── modules/                        # Tier 2 - Kept indefinitely
│   ├── alignment_short_read_1.0.0.sif
│   ├── variant_calling_1.0.0.sif
│   └── [...]
│
├── overlays/                       # Tier 3A - 30-day TTL
│   ├── base+star_2.7.11b.img      # overlay: base + specific STAR version
│   ├── alignment+rsem.img          # overlay: alignment_short_read + RSEM
│   └── [...]
│
├── microservices/                  # Tier 3B - 30-day TTL
│   ├── star_2.7.11b.sif
│   ├── rsem_1.3.3.sif
│   └── [...]
│
└── custom/                         # Tier 3C - 7-day TTL
    ├── user_123_rnaseq_v1.sif
    └── [...]
```

### Cache Lookup Logic

```python
def find_container(tools_needed):
    """
    1. Check cache index (Redis/SQLite): Have we built this exact combination?
    2. Check base/modules: Do we have pre-built container with these tools?
    3. Check overlays: Do we have overlay with these additions?
    4. Check microservices: Do we have individual tool containers?
    5. Return: Cache hit path OR None (need to build)
    """
    
    # Create fingerprint of required tools
    fingerprint = hash(frozenset([(t.name, t.version) for t in tools_needed]))
    
    # Query cache index
    cached = cache_index.get(fingerprint)
    if cached and Path(cached).exists():
        cache_index.update_last_used(fingerprint)  # Extend TTL
        return cached
    
    # Check domain modules
    for module in domain_modules:
        if module.has_all_tools(tools_needed):
            return module.path
    
    return None  # Need to build
```

### Cache Management

```python
class CacheManager:
    """Manages container cache lifecycle"""
    
    def cleanup_expired(self):
        """Remove containers past their TTL"""
        now = time.time()
        
        # Overlays: 30-day TTL
        for overlay in Path("/scratch/container_cache/overlays").glob("*.img"):
            if now - overlay.stat().st_mtime > 30 * 86400:
                overlay.unlink()
        
        # Microservices: 30-day TTL
        for sif in Path("/scratch/container_cache/microservices").glob("*.sif"):
            if now - sif.stat().st_mtime > 30 * 86400:
                sif.unlink()
        
        # Custom: 7-day TTL
        for sif in Path("/scratch/container_cache/custom").glob("*.sif"):
            if now - sif.stat().st_mtime > 7 * 86400:
                sif.unlink()
    
    def promote_to_module(self, container_path, usage_count):
        """
        If a custom container is used >10 times, promote to domain module.
        """
        if usage_count >= 10:
            # Move to modules/ and update registry
            target = Path("/scratch/container_cache/modules") / container_path.name
            shutil.move(container_path, target)
            module_registry.add(target)
```

---

## 5. User Experience Flow

### Example 1: Standard Workflow (Cache Hit)

```
User: "RNA-seq differential expression analysis"

AI Agent:
  → Workflow design: FastQC → STAR → featureCounts → DESeq2
  → Container check: alignment_short_read_1.0.0.sif has all tools
  → Container strategy: Use existing (0 build time)
  → Workflow launch: Immediate

Total prep time: 0 seconds
```

### Example 2: Tool Version Customization

```
User: "RNA-seq with STAR 2.7.11b specifically (not the default 2.7.10a)"

AI Agent:
  → Workflow design: FastQC → STAR 2.7.11b → featureCounts → DESeq2
  → Container check: alignment_short_read_1.0.0.sif has STAR 2.7.10a (wrong version)
  → Strategy decision: Build microservice for STAR 2.7.11b
  
  Container Builder:
    → Check cache: microservices/star_2.7.11b.sif exists! (cache hit)
    → Total prep time: 0 seconds
  
  Workflow uses:
    - FastQC: alignment_short_read_1.0.0.sif
    - STAR: microservices/star_2.7.11b.sif (specific version)
    - featureCounts: alignment_short_read_1.0.0.sif
    - DESeq2: alignment_short_read_1.0.0.sif
  
  → Workflow launch: Immediate
```

### Example 3: Custom Tool Addition

```
User: "RNA-seq but also run RSEM for isoform quantification"

AI Agent:
  → Workflow design: FastQC → STAR → featureCounts → DESeq2 → RSEM
  → Container check: alignment_short_read_1.0.0.sif lacks RSEM
  → Strategy decision: Create overlay with RSEM
  
  Container Builder:
    → Check cache: No overlay with "alignment_short_read + rsem"
    → Build new overlay:
        Base: alignment_short_read_1.0.0.sif
        Add: rsem=1.3.3
        Time: 3 minutes
        Output: overlays/alignment+rsem_abc123.img
  
  User notification: "Building container with RSEM... ETA 3 minutes"
  
  → After 3 min: Workflow launches
  → Overlay cached for future "STAR + RSEM" requests
```

### Example 4: User Custom Script

```
User: "RNA-seq with my custom normalization script (uploads custom_norm.R)"

AI Agent:
  → Workflow design: FastQC → STAR → featureCounts → custom_norm.R → plots
  → Container check: No container has custom_norm.R (obviously)
  → Strategy decision: JIT build with user script
  
  Container Builder:
    → Analyze custom_norm.R dependencies: Needs DESeq2, BiocGenerics, additional packages
    → Build strategy: Create microservice with script + deps
    → Build time: 5 minutes
    
  User notification: "Preparing container with your custom script... ETA 5 minutes"
  
  → After 5 min: Workflow launches
  → Container cached as custom/user123_rnaseq_custom_abc.sif (7-day TTL)
```

### Example 5: Complex Novel Workflow

```
User: "I want to combine RNA-seq, ChIP-seq, and ATAC-seq data to find active enhancers"

AI Agent:
  → Workflow design:
      RNA branch: STAR → featureCounts → DE analysis
      ChIP branch: Bowtie2 → MACS2 → Peak calling
      ATAC branch: Bowtie2 → MACS2 → Accessibility
      Integration: Custom R script for enhancer prediction
  
  → Container strategy:
      - RNA processes: alignment_short_read_1.0.0.sif (cache hit)
      - ChIP processes: peak_calling_1.0.0.sif (cache hit)
      - ATAC processes: peak_calling_1.0.0.sif (cache hit)
      - Integration: Build custom container with Bioconductor packages
  
  Container Builder:
    → Only integration step needs new container
    → Build time: 8 minutes
  
  User notification: "3/4 containers ready. Building integration container... ETA 8 minutes"
  
  → After 8 min: Workflow launches
  → 75% of processes start immediately, integration step waits for container
```

---

## 6. Implementation Phases

### Phase 1: Foundation (Weeks 1-4) - CURRENT
- ✅ Reuse existing monolithic Snakemake containers
- ✅ Prove Nextflow works with current containers
- Focus: Workflow translation, not container optimization yet

### Phase 2: Modular Containers (Weeks 5-8)
- Build Tier 2 domain modules (alignment, variant_calling, peak_calling, etc.)
- Migrate from monolithic → modular
- Test: Does module composition work as expected?

### Phase 3: Dynamic Container Support (Weeks 9-12)
- Implement overlay system
- Implement microservice builder
- Implement JIT build for custom containers
- Integrate with AI container strategy agent

### Phase 4: Caching & Optimization (Weeks 13-16)
- Deploy cache management system
- Implement cache promotion (popular customs → modules)
- Optimize build times (parallel builds, pre-warming)
- Monitor cache hit rates

---

## 7. Technical Considerations

### Singularity Overlay Limitations

**Pros**:
- ✅ Fast: Add tools without full rebuild
- ✅ Efficient: Only store delta
- ✅ Flexible: Multiple overlays from same base

**Cons**:
- ⚠️ Size limit: Overlay has max size (2-4 GB typical)
- ⚠️ Persistence: Overlay must be writable (not always possible on shared filesystems)
- ⚠️ Complexity: Two files to manage (base + overlay)

**Alternative: SIF with Sandbox**:
```bash
# Convert SIF to writable sandbox
singularity build --sandbox temp_sandbox/ base.sif

# Add tools to sandbox
singularity exec --writable temp_sandbox/ micromamba install new_tool

# Convert back to SIF
singularity build custom.sif temp_sandbox/

# Cleanup sandbox
rm -rf temp_sandbox/
```

**Recommendation**: 
- Use overlays for **quick experiments** (< 30 min lifetime)
- Use sandbox→SIF for **cacheable containers** (> 30 min lifetime)

### Build Time Optimization

**Parallel Builds**:
- Microservices can build in parallel (4-8 concurrent builds)
- Each build takes 3-5 minutes
- 8 microservices = 5 minutes total (not 40 minutes)

**Pre-warming**:
```python
# During low-usage periods (nights/weekends), pre-build popular combinations
def prewarm_cache():
    popular_combinations = [
        ("star_2.7.11b", "rsem_1.3.3"),
        ("salmon_1.10.0", "kallisto_0.48.0"),
        ("macs2_2.2.9.1", "macs3_3.0.0"),
        # ...
    ]
    
    for combo in popular_combinations:
        if not cache_exists(combo):
            build_microservices(combo)  # Background job
```

**Smart Queuing**:
- If build takes > 5 minutes, queue workflow for later execution
- User gets estimate: "Your workflow will start in ~12 minutes"
- User can choose: Wait now OR submit and get notified when ready

### Storage Management

**Estimated Storage Needs**:
```
Tier 1 (base): 2 GB × 1 = 2 GB
Tier 2 (modules): 3-5 GB × 10 modules = 40 GB
Tier 3 (dynamic):
  - Overlays: 500 MB × 50 cached = 25 GB
  - Microservices: 300 MB × 100 cached = 30 GB
  - Custom: 2 GB × 20 active = 40 GB

Total: ~140 GB (vs 1.2 TB for Snakemake per-user duplication)
```

**Garbage Collection**:
- Daily cleanup of expired containers
- Weekly analysis of cache hit rates
- Monthly promotion of popular customs to modules

---

## 8. Questions & Design Decisions

### Q1: Should we use Docker or Singularity for builds?

**Recommendation: Singularity (SIF)**
- ✅ Already using Singularity in Snakemake
- ✅ HPC-friendly (rootless, no daemon)
- ✅ Compatible with SLURM
- ✅ Can convert Docker images: `singularity pull docker://image`

**Docker**: Only if targeting cloud-native Kubernetes deployment (future)

---

### Q2: Who builds containers? User-facing API or backend service?

**Recommendation: Backend Service (Transparent to User)**

```
User submits query → AI agent generates workflow
                   ↓
    AI checks if containers exist
                   ↓
    If missing: Build in background (parallel to workflow generation)
                   ↓
    User sees: "Preparing environment... ETA X minutes"
                   ↓
    When ready: Workflow launches automatically
```

**Why NOT user-facing**:
- ❌ User shouldn't need to understand containers
- ❌ "Can you build me a container with STAR 2.7.11b?" is too technical
- ✅ "Run RNA-seq with latest STAR" → AI handles container selection/building

---

### Q3: Should custom containers be user-specific or shared?

**Recommendation: Shared Cache with Privacy Controls**

**Shared**:
- User A builds "STAR 2.7.11b" → cached
- User B requests "STAR 2.7.11b" → cache hit (instant)
- Benefit: 10× fewer builds, faster for all users

**Privacy**:
- User uploads `my_secret_algorithm.R` → private container (not shared)
- Flag: `private=True` → container stored in user's directory only
- Benefit: Protect proprietary code while sharing common tools

---

### Q4: What if container build fails?

**Failure Handling**:
```python
def handle_build_failure(build_spec, error):
    # 1. Log error for debugging
    logger.error(f"Container build failed: {error}")
    
    # 2. Fallback strategies
    if build_spec.strategy == 'microservice':
        # Try with broader version constraint
        return retry_with_relaxed_version(build_spec)
    
    elif build_spec.strategy == 'overlay':
        # Fall back to full custom build
        return build_custom_container(build_spec)
    
    elif build_spec.strategy == 'jit_build':
        # No fallback, notify user
        notify_user(f"Cannot build container: {error}")
        suggest_alternatives(build_spec)
        return None
    
# User gets clear message:
# "Unable to install tool XYZ version 1.2.3. Options:"
#   1. Use version 1.2.2 (available)"
#   2. Use alternative tool ABC"
#   3. Contact support with error details"
```

---

### Q5: How do we handle tool version conflicts?

**Scenario**: User wants STAR 2.7.11b + Salmon 1.9.0, but they have conflicting dependencies

**Solution: Process-Level Container Assignment**
```groovy
process STAR_ALIGN {
    container "microservices/star_2.7.11b.sif"  // STAR-specific container
}

process SALMON_QUANT {
    container "microservices/salmon_1.9.0.sif"  // Salmon-specific container
}

// No conflict! Each process has its own isolated environment
```

**Benefit**: Microservice architecture prevents dependency conflicts entirely

---

## 9. Example Container Definitions

### Microservice Example (STAR specific version)

```singularity
# microservices/star_2.7.11b.def
Bootstrap: docker
From: mambaorg/micromamba:1.5.8

%labels
    tool="STAR"
    version="2.7.11b"
    category="alignment"
    type="microservice"

%post
    micromamba install -y -n base -c conda-forge -c bioconda \
        star=2.7.11b \
        samtools=1.17
    
    micromamba clean --all --yes

%runscript
    exec STAR "$@"

%test
    STAR --version
```

**Build**: 3-4 minutes  
**Size**: ~450 MB  
**Usage**: Single-purpose, version-pinned

---

### Overlay Example (Add RSEM to alignment module)

```bash
# Create 2GB overlay
singularity overlay create --size 2048 /scratch/cache/overlays/alignment+rsem.img

# Install RSEM into overlay
singularity exec \
    --overlay /scratch/cache/overlays/alignment+rsem.img:rw \
    /scratch/cache/modules/alignment_short_read_1.0.0.sif \
    micromamba install -y -n base -c bioconda rsem=1.3.3

# Use in Nextflow
process RSEM_QUANT {
    containerOptions "--overlay /scratch/cache/overlays/alignment+rsem.img"
    container "/scratch/cache/modules/alignment_short_read_1.0.0.sif"
    
    script:
    """
    rsem-calculate-expression ...
    """
}
```

**Build**: 2-3 minutes  
**Size**: Base (4 GB) + Overlay (500 MB) = 4.5 GB effective  
**Usage**: Add missing tool to existing module

---

### Custom Container with User Script

```singularity
# custom/user123_normalization.def
Bootstrap: localimage
From: /scratch/cache/modules/alignment_short_read_1.0.0.sif

%files
    /home/user123/custom_normalize.R /opt/scripts/custom_normalize.R

%post
    # Install dependencies for user script
    micromamba install -y -n base -c conda-forge -c bioconda \
        r-optparse \
        r-jsonlite \
        bioconductor-edger \
        bioconductor-limma
    
    # Make script executable
    chmod +x /opt/scripts/custom_normalize.R
    
    micromamba clean --all --yes

%environment
    export PATH="/opt/scripts:$PATH"

%runscript
    exec /opt/scripts/custom_normalize.R "$@"
```

**Build**: 5-8 minutes (R package compilation)  
**Size**: ~5 GB  
**Usage**: User-provided analysis code + dependencies

---

## 10. Recommendations Summary

### ✅ Recommended Approach

**Multi-Tier Strategy**:
1. **Base + Domain Modules** (Tier 1-2): Pre-built, cover 90% of use cases
2. **Microservices** (Tier 3B): Fast builds (3-5 min), version-specific tools
3. **Overlays** (Tier 3A): Quick additions (2-3 min), temporary experiments
4. **Custom JIT** (Tier 3C): Full builds (10-30 min), complex requirements

**Implementation Priority**:
1. **Phase 2** (Weeks 5-8): Build domain modules
2. **Phase 3** (Weeks 9-12): Implement microservice builder
3. **Phase 3** (Weeks 9-12): Implement overlay support
4. **Phase 4** (Weeks 13-16): Implement full JIT custom builds

**Caching Strategy**:
- Shared cache with 30-day TTL for microservices/overlays
- Private containers for user-uploaded code
- Promote popular combinations to domain modules

### ⚠️ Concerns to Address

**Build Time**:
- Most builds: 2-5 minutes (acceptable)
- Complex builds: 10-30 minutes (need queueing system)
- Solution: Smart queuing + user notifications

**Storage Management**:
- Estimated 140 GB for cache (manageable)
- Need aggressive garbage collection
- Monitor cache hit rates to optimize

**Failure Handling**:
- Container builds can fail (dependency conflicts, network issues)
- Need robust fallback strategies
- Clear user communication when builds fail

### 🚀 Killer Features This Enables

1. **"Use latest version of tool X"** → AI builds microservice automatically
2. **"Run with my custom script"** → AI integrates user code seamlessly
3. **"Compare 3 aligners"** → AI creates microservices for each, runs in parallel
4. **"I need tool from GitHub repo"** → AI clones, builds, containerizes
5. **"Exactly reproduce paper X's pipeline"** → AI pins exact tool versions

This strategy makes the platform **truly flexible** while maintaining:
- Fast response times (cache hits are instant)
- Storage efficiency (shared cache, not per-user duplication)
- User simplicity (AI handles complexity)
- Production stability (base modules are immutable)

---

## 11. Next Steps

### Immediate (This Week)
- ✅ Document this strategy
- ✅ Validate with team
- Get approval for storage allocation (~200 GB for container cache)

### Phase 2 (Weeks 5-8)
- Build first domain module (alignment_short_read)
- Test process-level container assignment in Nextflow
- Validate storage and performance

### Phase 3 (Weeks 9-12)
- Implement microservice builder
- Integrate with AI container strategy agent
- Beta test with 3-5 users

### Phase 4 (Weeks 13-16)
- Deploy full caching system
- Implement garbage collection
- Launch to all users

---

**Status**: Strategy Documented - Awaiting Implementation  
**Expected Impact**: Enables truly dynamic, user-driven pipeline customization  
**Risk Level**: Medium (new technology, but well-scoped)  
**Recommendation**: ✅ Proceed with phased implementation

