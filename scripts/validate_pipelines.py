#!/usr/bin/env python3
"""
Pre-flight validation script for BioPipelines
Checks all prerequisites before pipeline execution
"""

import os
import sys
from pathlib import Path
import yaml

def check_file(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  ✓ {description}: {path} ({size:,} bytes)")
        return True
    else:
        print(f"  ✗ {description} MISSING: {path}")
        return False

def check_dir(path, description):
    """Check if a directory exists"""
    if os.path.isdir(path):
        count = len(os.listdir(path))
        print(f"  ✓ {description}: {path} ({count} items)")
        return True
    else:
        print(f"  ✗ {description} MISSING: {path}")
        return False

def validate_rna_seq():
    """Validate RNA-seq pipeline prerequisites"""
    print("\n═══ RNA-seq Pipeline ═══")
    config = yaml.safe_load(open("pipelines/rna_seq/differential_expression/config.yaml"))
    
    issues = []
    
    # Check reference files
    if not check_file(config["reference"]["genome"], "Reference genome"):
        issues.append("RNA-seq reference genome missing")
    if not check_file(config["reference"]["gtf"], "GTF annotation"):
        issues.append("RNA-seq GTF missing")
    if not check_dir(config["star_index"], "STAR index"):
        issues.append("RNA-seq STAR index missing")
    
    # Check sample files
    all_samples = config["samples"]["treatment"] + config["samples"]["control"]
    for sample in all_samples:
        r1 = f"/scratch/sdodl001/BioPipelines/data/raw/rna_seq/{sample}_R1.fastq.gz"
        r2 = f"/scratch/sdodl001/BioPipelines/data/raw/rna_seq/{sample}_R2.fastq.gz"
        if not check_file(r1, f"Sample {sample} R1"):
            issues.append(f"RNA-seq sample {sample} R1 missing")
        if not check_file(r2, f"Sample {sample} R2"):
            issues.append(f"RNA-seq sample {sample} R2 missing")
    
    return issues

def validate_atac_seq():
    """Validate ATAC-seq pipeline prerequisites"""
    print("\n═══ ATAC-seq Pipeline ═══")
    config = yaml.safe_load(open("pipelines/atac_seq/accessibility_analysis/config.yaml"))
    
    issues = []
    
    # Check reference files
    ref = config["reference"]["genome"]
    if not check_file(ref, "Reference genome"):
        issues.append("ATAC-seq reference genome missing")
    
    # Check bowtie2 index
    ref_base = ref.replace(".fa", "")
    index_files = [f"{ref_base}.{ext}" for ext in ["1.bt2", "2.bt2", "3.bt2", "4.bt2", "rev.1.bt2", "rev.2.bt2"]]
    for idx in index_files:
        if not os.path.exists(idx):
            print(f"  ✗ Bowtie2 index missing: {idx}")
            issues.append("ATAC-seq Bowtie2 index incomplete")
            break
    else:
        print(f"  ✓ Bowtie2 index: {ref_base}.*.bt2")
    
    # Check sample files
    for sample in config["samples"]:
        r1 = f"/scratch/sdodl001/BioPipelines/data/raw/atac_seq/{sample}_R1.fastq.gz"
        r2 = f"/scratch/sdodl001/BioPipelines/data/raw/atac_seq/{sample}_R2.fastq.gz"
        if not check_file(r1, f"Sample {sample} R1"):
            issues.append(f"ATAC-seq sample {sample} R1 missing")
        if not check_file(r2, f"Sample {sample} R2"):
            issues.append(f"ATAC-seq sample {sample} R2 missing")
    
    return issues

def validate_chip_seq():
    """Validate ChIP-seq pipeline prerequisites"""
    print("\n═══ ChIP-seq Pipeline ═══")
    config = yaml.safe_load(open("pipelines/chip_seq/peak_calling/config.yaml"))
    
    issues = []
    
    # Check reference files
    ref = config["reference"]["genome"]
    if not check_file(ref, "Reference genome"):
        issues.append("ChIP-seq reference genome missing")
    
    # Check BWA index
    for ext in [".amb", ".ann", ".bwt", ".pac", ".sa"]:
        if not check_file(ref + ext, f"BWA index {ext}"):
            issues.append(f"ChIP-seq BWA index {ext} missing")
    
    # Check sample files
    all_samples = config["samples"] + [config["input_control"]]
    for sample in all_samples:
        r1 = f"/scratch/sdodl001/BioPipelines/data/raw/chip_seq/{sample}_R1.fastq.gz"
        r2 = f"/scratch/sdodl001/BioPipelines/data/raw/chip_seq/{sample}_R2.fastq.gz"
        if not check_file(r1, f"Sample {sample} R1"):
            issues.append(f"ChIP-seq sample {sample} R1 missing")
        if not check_file(r2, f"Sample {sample} R2"):
            issues.append(f"ChIP-seq sample {sample} R2 missing")
    
    return issues

def validate_dna_seq():
    """Validate DNA-seq pipeline prerequisites"""
    print("\n═══ DNA-seq Pipeline ═══")
    config = yaml.safe_load(open("pipelines/dna_seq/variant_calling/config.yaml"))
    
    issues = []
    
    # Check reference files
    ref = config["reference"]["genome"]
    if not check_file(ref, "Reference genome"):
        issues.append("DNA-seq reference genome missing")
    if not check_file(config["reference"]["known_sites"], "Known sites (dbSNP)"):
        issues.append("DNA-seq dbSNP file missing")
    
    # Check BWA index
    for ext in [".amb", ".ann", ".bwt", ".pac", ".sa"]:
        if not check_file(ref + ext, f"BWA index {ext}"):
            issues.append(f"DNA-seq BWA index {ext} missing")
    
    # Check sample files
    for sample in config["samples"]:
        r1 = f"/scratch/sdodl001/BioPipelines/data/raw/dna_seq/{sample}_R1.fastq.gz"
        r2 = f"/scratch/sdodl001/BioPipelines/data/raw/dna_seq/{sample}_R2.fastq.gz"
        if not check_file(r1, f"Sample {sample} R1"):
            issues.append(f"DNA-seq sample {sample} R1 missing")
        if not check_file(r2, f"Sample {sample} R2"):
            issues.append(f"DNA-seq sample {sample} R2 missing")
    
    return issues

def main():
    """Main validation function"""
    print("╔════════════════════════════════════════╗")
    print("║  BioPipelines Pre-flight Validation   ║")
    print("╚════════════════════════════════════════╝")
    
    os.chdir("/home/sdodl001_odu_edu/BioPipelines")
    
    all_issues = []
    
    all_issues.extend(validate_rna_seq())
    all_issues.extend(validate_atac_seq())
    all_issues.extend(validate_chip_seq())
    all_issues.extend(validate_dna_seq())
    
    print("\n" + "="*50)
    if all_issues:
        print("❌ VALIDATION FAILED")
        print(f"\nFound {len(all_issues)} issue(s):")
        for issue in all_issues:
            print(f"  • {issue}")
        sys.exit(1)
    else:
        print("✅ ALL CHECKS PASSED")
        print("\nAll pipelines are ready to run!")
        sys.exit(0)

if __name__ == "__main__":
    main()
