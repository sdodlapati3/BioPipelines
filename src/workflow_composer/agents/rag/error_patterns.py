"""
Error Pattern Database
======================

Common bioinformatics workflow errors and solutions.

Features:
- Pattern matching for error messages
- Context-aware solution suggestions
- Analysis type specific fixes
"""

import logging
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ErrorSolution:
    """A solution for an error pattern."""
    pattern: str
    pattern_regex: Optional[re.Pattern] = None
    cause: str = ""
    solution: str = ""
    category: str = "general"
    analysis_types: List[str] = field(default_factory=list)
    confidence: float = 0.8
    examples: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Compile regex pattern."""
        if self.pattern_regex is None and self.pattern:
            try:
                self.pattern_regex = re.compile(self.pattern, re.IGNORECASE)
            except re.error:
                # If pattern is not valid regex, use simple string matching
                self.pattern_regex = None
    
    def matches(self, error_text: str) -> Tuple[bool, float]:
        """
        Check if error text matches this pattern.
        
        Args:
            error_text: Error message to check
            
        Returns:
            Tuple of (matches, confidence)
        """
        if self.pattern_regex:
            match = self.pattern_regex.search(error_text)
            if match:
                return True, self.confidence
        elif self.pattern.lower() in error_text.lower():
            return True, self.confidence * 0.9
        
        return False, 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern": self.pattern,
            "cause": self.cause,
            "solution": self.solution,
            "category": self.category,
            "analysis_types": self.analysis_types,
            "confidence": self.confidence,
            "examples": self.examples,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorSolution":
        """Create from dictionary."""
        return cls(
            pattern=data.get("pattern", ""),
            cause=data.get("cause", ""),
            solution=data.get("solution", ""),
            category=data.get("category", "general"),
            analysis_types=data.get("analysis_types", []),
            confidence=data.get("confidence", 0.8),
            examples=data.get("examples", []),
        )


class ErrorPatternDB:
    """
    Database of error patterns and solutions.
    
    Matches error messages against known patterns and
    provides context-aware solutions.
    """
    
    # Default error patterns
    DEFAULT_PATTERNS = [
        # Memory errors
        {
            "pattern": r"Out of memory|OOM|Cannot allocate memory|MemoryError",
            "cause": "Process exceeded available memory",
            "solution": "Increase memory allocation in process directive: memory = '32 GB'",
            "category": "memory",
            "confidence": 0.95,
            "examples": ["STAR alignment ran out of memory", "Java heap space exceeded"],
        },
        {
            "pattern": r"SLURM.*exceeded.*memory|memory limit.*exceeded|Killed",
            "cause": "SLURM killed job due to memory limit",
            "solution": "Increase --max_memory parameter or add higher memory label to process",
            "category": "memory",
            "confidence": 0.9,
        },
        
        # Disk errors
        {
            "pattern": r"No space left on device|disk quota exceeded|ENOSPC",
            "cause": "Disk full or quota exceeded",
            "solution": "Clear temp files, increase scratch space, or use different work directory: -work-dir /scratch/work",
            "category": "disk",
            "confidence": 0.95,
        },
        {
            "pattern": r"Permission denied|EACCES",
            "cause": "Insufficient file permissions",
            "solution": "Check file permissions and ownership. Ensure write access to output directory.",
            "category": "permissions",
            "confidence": 0.85,
        },
        
        # Input errors
        {
            "pattern": r"No such file or directory|ENOENT|file not found",
            "cause": "Input file or reference not found",
            "solution": "Verify input file paths exist and are accessible. Check for typos in --input parameter.",
            "category": "input",
            "confidence": 0.9,
        },
        {
            "pattern": r"Empty input|No reads|No sequences found",
            "cause": "Input files are empty or have no valid sequences",
            "solution": "Check input files for content. Verify FASTQ format is correct.",
            "category": "input",
            "confidence": 0.85,
        },
        {
            "pattern": r"truncated|corrupt|Invalid.*format|Bad.*header",
            "cause": "Corrupted or truncated input files",
            "solution": "Re-download or regenerate input files. Check file integrity with md5sum.",
            "category": "input",
            "confidence": 0.9,
        },
        
        # Reference errors
        {
            "pattern": r"Index not found|genome index|reference.*not.*exist",
            "cause": "Reference genome index not found",
            "solution": "Build reference index first: use --genome parameter or provide --star_index/--bowtie2_index",
            "category": "reference",
            "confidence": 0.9,
            "analysis_types": ["rna-seq", "dna-seq", "chip-seq"],
        },
        {
            "pattern": r"Chromosome.*not found|contig.*not in|sequence.*missing",
            "cause": "Chromosome/contig names don't match between files",
            "solution": "Ensure consistent chromosome naming (e.g., 'chr1' vs '1'). Use matching reference and annotation.",
            "category": "reference",
            "confidence": 0.85,
        },
        
        # Container errors
        {
            "pattern": r"Container.*not found|docker.*pull.*failed|singularity.*error",
            "cause": "Container image not available",
            "solution": "Check internet connection for container pulling. Use -profile docker or -profile singularity.",
            "category": "container",
            "confidence": 0.9,
        },
        {
            "pattern": r"Image.*does not exist|manifest.*not found",
            "cause": "Container image tag not found",
            "solution": "Verify container version exists. Try using 'latest' tag or specific version.",
            "category": "container",
            "confidence": 0.85,
        },
        
        # SLURM errors
        {
            "pattern": r"SLURM.*TIMEOUT|DUE TO TIME LIMIT|time limit",
            "cause": "Job exceeded time limit",
            "solution": "Increase time limit in process directive: time = '24.h' or use higher time label",
            "category": "slurm",
            "confidence": 0.95,
        },
        {
            "pattern": r"QOSMax.*Limit|partition.*unavailable|submission.*failed",
            "cause": "SLURM resource limits or partition issue",
            "solution": "Check available partitions with 'sinfo'. Reduce resource requests or use different partition.",
            "category": "slurm",
            "confidence": 0.8,
        },
        
        # Tool-specific errors
        {
            "pattern": r"STAR.*fatal error|STAR.*EXITING",
            "cause": "STAR aligner encountered a fatal error",
            "solution": "Check genome index compatibility. Ensure sufficient memory (32GB+ for human). Verify annotation GTF format.",
            "category": "tool",
            "analysis_types": ["rna-seq"],
            "confidence": 0.9,
        },
        {
            "pattern": r"salmon.*index|salmon.*quant.*error",
            "cause": "Salmon quantification error",
            "solution": "Verify transcriptome index. Check read length compatibility with index k-mer size.",
            "category": "tool",
            "analysis_types": ["rna-seq"],
            "confidence": 0.85,
        },
        {
            "pattern": r"BWA.*error|bwa mem.*fail",
            "cause": "BWA alignment error",
            "solution": "Check reference index. Verify read format. Ensure paired-end files are properly matched.",
            "category": "tool",
            "analysis_types": ["dna-seq", "chip-seq"],
            "confidence": 0.85,
        },
        {
            "pattern": r"GATK.*error|HaplotypeCaller.*fail",
            "cause": "GATK tool error",
            "solution": "Check reference FASTA and dictionary files. Verify BAM file is sorted and indexed.",
            "category": "tool",
            "analysis_types": ["dna-seq"],
            "confidence": 0.85,
        },
        {
            "pattern": r"MACS2.*error|peak.*calling.*fail",
            "cause": "MACS2 peak calling error",
            "solution": "Check BAM files are properly sorted. Verify effective genome size parameter (-g).",
            "category": "tool",
            "analysis_types": ["chip-seq", "atac-seq"],
            "confidence": 0.85,
        },
        
        # Nextflow errors
        {
            "pattern": r"Missing.*param|required.*param.*not.*set",
            "cause": "Required parameter not provided",
            "solution": "Check pipeline documentation for required parameters. Provide missing parameters.",
            "category": "nextflow",
            "confidence": 0.9,
        },
        {
            "pattern": r"Channel.*empty|No matching.*input",
            "cause": "Input channel is empty - no files match pattern",
            "solution": "Check input file pattern. Verify files exist at specified location. Test glob pattern.",
            "category": "nextflow",
            "confidence": 0.85,
        },
        {
            "pattern": r"Process.*terminated|exit.*status.*[1-9]",
            "cause": "Process exited with non-zero status",
            "solution": "Check the .command.log and .command.err files in the work directory for details.",
            "category": "nextflow",
            "confidence": 0.7,
        },
    ]
    
    def __init__(self, patterns_file: str = None):
        """
        Initialize error pattern database.
        
        Args:
            patterns_file: Optional path to custom patterns YAML
        """
        self.patterns: List[ErrorSolution] = []
        
        # Load default patterns
        for pattern_data in self.DEFAULT_PATTERNS:
            self.patterns.append(ErrorSolution.from_dict(pattern_data))
        
        # Load custom patterns if provided
        if patterns_file:
            self.load_patterns(patterns_file)
    
    def load_patterns(self, patterns_file: str):
        """
        Load patterns from YAML file.
        
        Args:
            patterns_file: Path to patterns YAML
        """
        path = Path(patterns_file)
        if not path.exists():
            logger.warning(f"Patterns file not found: {patterns_file}")
            return
        
        try:
            data = yaml.safe_load(path.read_text())
            patterns = data.get("patterns", data) if isinstance(data, dict) else data
            
            for pattern_data in patterns:
                self.patterns.append(ErrorSolution.from_dict(pattern_data))
            
            logger.info(f"Loaded {len(patterns)} custom error patterns")
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
    
    def find_solution(self, error_text: str, 
                      analysis_type: str = None) -> Optional[ErrorSolution]:
        """
        Find best matching solution for an error.
        
        Args:
            error_text: Error message to analyze
            analysis_type: Optional analysis type for context
            
        Returns:
            Best matching ErrorSolution or None
        """
        matches = []
        
        for pattern in self.patterns:
            is_match, confidence = pattern.matches(error_text)
            
            if is_match:
                # Boost confidence for analysis-type specific patterns
                if analysis_type and pattern.analysis_types:
                    if analysis_type in pattern.analysis_types:
                        confidence *= 1.1
                    else:
                        confidence *= 0.8
                
                matches.append((pattern, confidence))
        
        if not matches:
            return None
        
        # Return highest confidence match
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[0][0]
    
    def find_all_solutions(self, error_text: str,
                           analysis_type: str = None,
                           limit: int = 5) -> List[Tuple[ErrorSolution, float]]:
        """
        Find all matching solutions ranked by confidence.
        
        Args:
            error_text: Error message to analyze
            analysis_type: Optional analysis type for context
            limit: Maximum solutions to return
            
        Returns:
            List of (ErrorSolution, confidence) tuples
        """
        matches = []
        
        for pattern in self.patterns:
            is_match, confidence = pattern.matches(error_text)
            
            if is_match:
                if analysis_type and pattern.analysis_types:
                    if analysis_type in pattern.analysis_types:
                        confidence *= 1.1
                    else:
                        confidence *= 0.8
                
                matches.append((pattern, min(confidence, 1.0)))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]
    
    def format_solution(self, error_text: str,
                        analysis_type: str = None) -> Optional[str]:
        """
        Get formatted solution text for an error.
        
        Args:
            error_text: Error message
            analysis_type: Analysis type context
            
        Returns:
            Formatted solution string or None
        """
        solution = self.find_solution(error_text, analysis_type)
        
        if not solution:
            return None
        
        parts = [
            f"**Error Category:** {solution.category}",
            f"**Likely Cause:** {solution.cause}",
            f"**Suggested Solution:** {solution.solution}",
        ]
        
        if solution.examples:
            parts.append(f"**Similar Errors:** {', '.join(solution.examples[:3])}")
        
        return "\n".join(parts)
    
    def add_pattern(self, pattern: ErrorSolution):
        """Add a new pattern to the database."""
        self.patterns.append(pattern)
    
    def get_patterns_by_category(self, category: str) -> List[ErrorSolution]:
        """Get all patterns in a category."""
        return [p for p in self.patterns if p.category == category]
    
    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(p.category for p in self.patterns))
    
    def export_patterns(self, output_file: str):
        """Export patterns to YAML file."""
        data = {
            "patterns": [p.to_dict() for p in self.patterns]
        }
        
        Path(output_file).write_text(yaml.dump(data, default_flow_style=False))
        logger.info(f"Exported {len(self.patterns)} patterns to {output_file}")
