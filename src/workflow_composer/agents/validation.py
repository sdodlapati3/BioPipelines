"""
Intelligent Validation Layer
============================

Provides multi-stage validation for agent responses:
1. Context Extraction - Understand what user really wants
2. Tool Result Validation - Check if results match intent
3. Cross-Source Verification - Validate with multiple methods
4. Confidence Scoring - Rate response reliability

This layer ensures the agent doesn't give misleading responses.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from .intent.context import ConversationContext as UnifiedContext

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for validated responses."""
    HIGH = "high"        # All checks passed, data verified
    MEDIUM = "medium"    # Most checks passed, some uncertainty
    LOW = "low"          # Significant uncertainty, needs clarification
    UNKNOWN = "unknown"  # Cannot determine, user should verify


@dataclass
class UserIntent:
    """Parsed user intent from conversation."""
    action: str                          # What they want to do (search, scan, download, etc.)
    data_type: Optional[str] = None      # RNA-seq, methylation, ChIP-seq, etc.
    organism: Optional[str] = None       # human, mouse, etc.
    tissue: Optional[str] = None         # brain, liver, etc.
    condition: Optional[str] = None      # cancer, diabetes, etc.
    target: Optional[str] = None         # Gene, protein target
    keywords: List[str] = field(default_factory=list)
    raw_query: str = ""


@dataclass 
class ValidationResult:
    """Result of validating a tool response."""
    is_valid: bool
    confidence: ConfidenceLevel
    original_response: str
    validated_response: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationContext:
    """
    Lightweight validation context for ResponseValidator.
    
    NOTE: For full conversation management, use the unified ConversationContext
    from workflow_composer.agents.intent.context instead.
    
    This class only handles intent extraction for validation purposes.
    """
    
    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, str]] = []
        self.current_intent: Optional[UserIntent] = None
        self.max_history = max_history
        self.active_queries: List[str] = []  # Recent search queries
        self.selected_datasets: List[str] = []  # Datasets user showed interest in
    
    def add_message(self, role: str, content):
        """Add a message to history and update context."""
        # Handle Gradio's list format for multimodal content
        if isinstance(content, list):
            # Extract text from list of content items
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
            content = ' '.join(text_parts) if text_parts else ''
        
        if not isinstance(content, str):
            content = str(content) if content else ''
        
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Extract intent from user messages
        if role == "user" and content:
            new_intent = self._extract_intent(content)
            
                # IMPORTANT: Preserve prior intent if new query is generic
            # (e.g., "what data is available" shouldn't clear "human brain methylation")
            if new_intent.action in ["scan", "unknown"] and not new_intent.data_type:
                # Generic query - preserve prior context
                if self.current_intent:
                    new_intent.data_type = self.current_intent.data_type
                    new_intent.organism = self.current_intent.organism
                    new_intent.tissue = self.current_intent.tissue
                    new_intent.condition = self.current_intent.condition
            
            self.current_intent = new_intent
            
            # Track queries
            if new_intent.action in ["search", "find"]:
                self.active_queries.append(content)
                if len(self.active_queries) > 5:
                    self.active_queries.pop(0)
    
    def _extract_intent(self, message: str) -> UserIntent:
        """Extract structured intent from user message."""
        message_lower = message.lower()
        
        # Detect action
        action = "unknown"
        action_patterns = {
            "search": r"search|find|look for|query|discover",
            "scan": r"scan|check|show|list|what.*have|available",
            "download": r"download|get|fetch|add",
            "analyze": r"analyze|run|process|compare",
            "help": r"help|how|what can",
        }
        for act, pattern in action_patterns.items():
            if re.search(pattern, message_lower):
                action = act
                break
        
        # Detect data type / assay
        data_type = None
        assay_patterns = {
            "methylation": r"methyl|wgbs|bisulfite|5mc|dna methyl|rrbs|cpg|cpg island",
            "rna-seq": r"rna[\s\-]?seq|transcriptom|gene express|mrna",
            "scrna-seq": r"scrna|single[\s\-]?cell|10x|smart[\s\-]?seq",
            "chip-seq": r"chip[\s\-]?seq|histone|h3k|chromatin immuno",
            "atac-seq": r"atac[\s\-]?seq|chromatin access|open chromatin",
            "wgs": r"wgs|whole[\s\-]?genome seq|dna[\s\-]?seq(?!uencing)|exome",
            "hic": r"hi[\s\-]?c|chromatin interact|3d genome",
        }
        for dtype, pattern in assay_patterns.items():
            if re.search(pattern, message_lower):
                data_type = dtype
                break
        
        # Detect organism
        organism = None
        organism_patterns = {
            "human": r"human|homo sapiens|hg38|grch38|grch37|hg19",
            "mouse": r"mouse|mus musculus|mm10|mm39|grcm",
            "rat": r"\brat\b|rattus|rn6|rn7",
            "zebrafish": r"zebrafish|danio|grcz|zv",
            "drosophila": r"drosophila|fly|dm6|bdgp",
            "yeast": r"yeast|saccharomyces|cerevisiae",
        }
        for org, pattern in organism_patterns.items():
            if re.search(pattern, message_lower):
                organism = org
                break
        
        # Detect tissue
        tissue = None
        tissue_patterns = {
            "brain": r"brain|neuro|cortex|hippocampus|cerebr",
            "liver": r"liver|hepat",
            "heart": r"heart|cardi",
            "lung": r"lung|pulmon",
            "kidney": r"kidney|renal",
            "blood": r"blood|pbmc|leukocyte|lymphocyte|monocyte",
            "stem cell": r"stem cell|ips|esc|pluripotent",
        }
        
        # Detect disease/condition (separate from tissue)
        condition = None
        condition_patterns = {
            "cancer": r"cancer|tumor|tumour|carcinoma|malignant|oncolog|glioma|glioblastoma|melanoma|lymphoma|leukemia",
            "alzheimer": r"alzheimer|dementia|neurodegenerat",
            "diabetes": r"diabet|insulin",
            "infection": r"infect|viral|bacteria|pathogen",
        }
        for tis, pattern in tissue_patterns.items():
            if re.search(pattern, message_lower):
                tissue = tis
                break
        
        # Detect condition/disease
        for cond, pattern in condition_patterns.items():
            if re.search(pattern, message_lower):
                condition = cond
                break
        
        # Extract keywords (nouns and important terms)
        keywords = []
        # Simple keyword extraction - words that seem important
        important_words = re.findall(r'\b[a-z]{4,}\b', message_lower)
        stopwords = {'that', 'this', 'what', 'have', 'with', 'from', 'want', 'need', 
                     'would', 'could', 'should', 'please', 'help', 'find', 'search',
                     'look', 'show', 'data', 'files', 'samples'}
        keywords = [w for w in important_words if w not in stopwords][:10]
        
        return UserIntent(
            action=action,
            data_type=data_type,
            organism=organism,
            tissue=tissue,
            condition=condition,
            keywords=keywords,
            raw_query=message
        )
    
    def get_context_summary(self) -> str:
        """Get a summary of current context for LLM."""
        parts = []
        
        if self.current_intent:
            intent = self.current_intent
            if intent.data_type:
                parts.append(f"Data type: {intent.data_type}")
            if intent.organism:
                parts.append(f"Organism: {intent.organism}")
            if intent.tissue:
                parts.append(f"Tissue: {intent.tissue}")
            if intent.condition:
                parts.append(f"Condition: {intent.condition}")
        
        if self.active_queries:
            parts.append(f"Recent queries: {', '.join(self.active_queries[-3:])}")
        
        if self.selected_datasets:
            parts.append(f"Selected datasets: {', '.join(self.selected_datasets[-3:])}")
        
        return "; ".join(parts) if parts else "No specific context"
    
    def matches_intent(self, data_info: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if data info matches the current intent.
        
        Returns:
            (matches, list of mismatches)
        """
        if not self.current_intent:
            return True, []
        
        mismatches = []
        intent = self.current_intent
        
        # Check data type
        if intent.data_type:
            data_assay = data_info.get("assay_type", "").lower()
            if data_assay and intent.data_type.lower() not in data_assay:
                mismatches.append(f"Expected {intent.data_type}, found {data_assay or 'unknown'}")
        
        # Check organism
        if intent.organism:
            data_org = data_info.get("organism", "").lower()
            if data_org and intent.organism.lower() not in data_org:
                mismatches.append(f"Expected {intent.organism}, found {data_org or 'unknown'}")
        
        # Check tissue
        if intent.tissue:
            data_tissue = data_info.get("tissue", "").lower()
            if data_tissue and intent.tissue.lower() not in data_tissue:
                mismatches.append(f"Expected {intent.tissue}, found {data_tissue or 'unknown'}")
        
        return len(mismatches) == 0, mismatches


class DataMetadataManager:
    """
    Manages metadata.json files for data directories.
    Tracks organism, tissue, assay type, source, etc.
    """
    
    METADATA_FILE = "metadata.json"
    
    def __init__(self, base_data_path: Path = None):
        self.base_path = base_data_path or Path("/scratch/sdodl001/BioPipelines/data")
    
    def read_metadata(self, data_path: Path) -> Optional[Dict[str, Any]]:
        """Read metadata from a data directory."""
        metadata_file = data_path / self.METADATA_FILE
        
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
        
        # Try parent directory
        parent_metadata = data_path.parent / self.METADATA_FILE
        if parent_metadata.exists():
            try:
                with open(parent_metadata) as f:
                    return json.load(f)
            except:
                pass
        
        return None
    
    def write_metadata(self, data_path: Path, metadata: Dict[str, Any]) -> bool:
        """Write metadata to a data directory."""
        metadata_file = data_path / self.METADATA_FILE
        
        try:
            # Merge with existing metadata
            existing = self.read_metadata(data_path) or {}
            existing.update(metadata)
            existing["last_updated"] = self._timestamp()
            
            with open(metadata_file, 'w') as f:
                json.dump(existing, f, indent=2)
            
            logger.info(f"Wrote metadata to {metadata_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to write metadata to {metadata_file}: {e}")
            return False
    
    def infer_metadata_from_path(self, data_path: Path) -> Dict[str, Any]:
        """Infer metadata from directory structure and filenames."""
        metadata = {}
        path_str = str(data_path).lower()
        
        # Infer assay type from path
        assay_hints = {
            "methylation": ["methylation", "wgbs", "rrbs", "bisulfite"],
            "rna-seq": ["rna_seq", "rna-seq", "rnaseq", "transcriptome"],
            "chip-seq": ["chip_seq", "chip-seq", "chipseq"],
            "atac-seq": ["atac_seq", "atac-seq", "atacseq"],
            "scrna-seq": ["scrna", "single_cell", "10x"],
            "hic": ["hic", "hi-c"],
            "wgs": ["wgs", "dna_seq", "exome"],
        }
        
        for assay, hints in assay_hints.items():
            if any(h in path_str for h in hints):
                metadata["assay_type"] = assay
                break
        
        # Check for GSE/ENCSR IDs in path
        gse_match = re.search(r'(GSE\d+)', str(data_path), re.IGNORECASE)
        if gse_match:
            metadata["source_id"] = gse_match.group(1).upper()
            metadata["source"] = "GEO"
        
        encsr_match = re.search(r'(ENCSR[A-Z0-9]+)', str(data_path), re.IGNORECASE)
        if encsr_match:
            metadata["source_id"] = encsr_match.group(1).upper()
            metadata["source"] = "ENCODE"
        
        return metadata
    
    def _timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()


class ResponseValidator:
    """
    Validates tool responses against user intent.
    Uses LLM for complex validation when needed.
    """
    
    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.model = model
        self.context = ValidationContext()  # Lightweight validation context
        self.metadata_manager = DataMetadataManager()
    
    def validate_scan_result(
        self,
        scan_result: Dict[str, Any],
        original_response: str,
    ) -> ValidationResult:
        """
        Validate a scan_data result against user intent.
        """
        issues = []
        suggestions = []
        confidence = ConfidenceLevel.HIGH
        
        # Get the scanned path
        scan_path = Path(scan_result.get("path", ""))
        sample_count = scan_result.get("count", 0)
        samples = scan_result.get("samples", [])
        
        # Check 1: Read metadata if available
        metadata = self.metadata_manager.read_metadata(scan_path)
        inferred_meta = self.metadata_manager.infer_metadata_from_path(scan_path)
        combined_meta = {**(inferred_meta or {}), **(metadata or {})}
        
        # Check 2: Validate against intent
        intent = self.context.current_intent
        if intent and (intent.data_type or intent.organism or intent.tissue):
            # User has specific requirements
            
            # If scanning a generic path (not type-specific), warn about mixed data
            path_str = str(scan_path).lower()
            is_typed_folder = any(t in path_str for t in 
                ['methylation', 'rna_seq', 'chip_seq', 'atac_seq', 'scrna', 'hic', 'wgs'])
            
            if not is_typed_folder and sample_count > 0:
                # Scanning top-level folder - contains mixed data types!
                # Build description of what user asked for
                asked_for_parts = []
                if intent.data_type:
                    asked_for_parts.append(intent.data_type)
                if intent.tissue:
                    asked_for_parts.append(intent.tissue)
                if intent.condition:
                    asked_for_parts.append(intent.condition)
                asked_for = ' '.join(asked_for_parts) if asked_for_parts else 'specific'
                
                issues.append(
                    f"⚠️ This folder contains ALL data types ({sample_count} samples total). "
                    f"You asked about {asked_for} data."
                )
                suggestions.append(
                    f"To see only {intent.data_type or 'relevant'} data, scan the specific subfolder:"
                )
                
                # Suggest specific paths - find the data/raw folder
                data_type = intent.data_type or ""
                folder_map = {
                    "methylation": "methylation",
                    "rna-seq": "rna_seq", 
                    "chip-seq": "chip_seq",
                    "atac-seq": "atac_seq",
                    "scrna-seq": "scrna_seq",
                    "hic": "hic",
                    "wgs": "dna_seq",
                }
                
                # Find the correct base path (avoid /data/data duplication)
                base_path = scan_path
                if 'data/raw' in str(scan_path):
                    # Already in data/raw, go up
                    base_path = scan_path.parent.parent
                elif str(scan_path).endswith('/data'):
                    base_path = scan_path
                
                if data_type in folder_map:
                    subfolder = folder_map[data_type]
                    suggestions.append(f"  `scan data in {base_path}/raw/{subfolder}`")
                else:
                    suggestions.append(f"  Check subfolders: {base_path}/raw/methylation/, rna_seq/, chip_seq/, etc.")
                
                confidence = ConfidenceLevel.LOW
            
            elif is_typed_folder:
                # Scanning a specific typed folder
                matches, mismatches = self.context.matches_intent(combined_meta)
                
                if not matches:
                    issues.extend(mismatches)
                    confidence = ConfidenceLevel.MEDIUM
                elif not combined_meta:
                    issues.append("Cannot verify data type - no metadata available")
                    confidence = ConfidenceLevel.MEDIUM
                    suggestions.append("Consider adding metadata.json with organism, tissue, assay_type")
        
        # Check 3: Analyze sample names for clues
        sample_analysis = self._analyze_sample_names(samples)
        if sample_analysis.get("warnings"):
            issues.extend(sample_analysis["warnings"])
            if confidence == ConfidenceLevel.HIGH:
                confidence = ConfidenceLevel.MEDIUM
        
        # Build validated response
        validated_response = self._build_validated_response(
            original_response, 
            scan_result,
            combined_meta,
            issues,
            suggestions,
            confidence
        )
        
        return ValidationResult(
            is_valid=confidence != ConfidenceLevel.LOW,
            confidence=confidence,
            original_response=original_response,
            validated_response=validated_response,
            issues=issues,
            suggestions=suggestions,
            metadata=combined_meta
        )
    
    def validate_search_result(
        self,
        search_result: Dict[str, Any],
        original_response: str,
    ) -> ValidationResult:
        """Validate a search_databases result."""
        issues = []
        confidence = ConfidenceLevel.HIGH
        
        results = search_result.get("results", [])
        query = search_result.get("query", "")
        
        # Check if results match the query intent
        intent = self.context.current_intent
        
        if not results:
            confidence = ConfidenceLevel.LOW
            issues.append(f"No results found for '{query}'")
        elif intent:
            # Check if results match expected organism/tissue/assay
            matching_results = []
            for r in results:
                matches = True
                if intent.organism and r.get("organism"):
                    if intent.organism.lower() not in r["organism"].lower():
                        matches = False
                if intent.data_type and r.get("assay"):
                    if intent.data_type.lower() not in r["assay"].lower():
                        matches = False
                if matches:
                    matching_results.append(r)
            
            if len(matching_results) < len(results):
                issues.append(
                    f"Some results may not match your criteria "
                    f"({len(matching_results)}/{len(results)} match)"
                )
                confidence = ConfidenceLevel.MEDIUM
        
        return ValidationResult(
            is_valid=True,
            confidence=confidence,
            original_response=original_response,
            validated_response=original_response,  # Search results are generally reliable
            issues=issues,
            suggestions=[],
            metadata={"query": query, "result_count": len(results)}
        )
    
    def _analyze_sample_names(self, samples: List[Any]) -> Dict[str, Any]:
        """Analyze sample names for patterns and issues."""
        warnings = []
        patterns = {}
        
        if not samples:
            return {"warnings": [], "patterns": {}}
        
        sample_names = []
        for s in samples:
            if hasattr(s, 'sample_id'):
                sample_names.append(s.sample_id)
            elif isinstance(s, dict):
                sample_names.append(s.get('sample_id', str(s)))
            else:
                sample_names.append(str(s))
        
        # Check for generic names
        generic_count = sum(1 for n in sample_names 
                          if re.match(r'^(sample|test|unknown|file)\d*', n.lower()))
        if generic_count > len(sample_names) * 0.5:
            warnings.append(
                f"{generic_count}/{len(sample_names)} samples have generic names - "
                "cannot determine data type from names alone"
            )
        
        # Look for assay hints in names
        assay_hints = {
            "methylation": r"meth|wgbs|bs-?seq|bisulf",
            "rna": r"rna|expr|transcr",
            "chip": r"chip|h3k|input|control",
            "atac": r"atac|dnase|access",
        }
        
        for assay, pattern in assay_hints.items():
            count = sum(1 for n in sample_names if re.search(pattern, n.lower()))
            if count > 0:
                patterns[assay] = count
        
        return {"warnings": warnings, "patterns": patterns}
    
    def _build_validated_response(
        self,
        original: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
        issues: List[str],
        suggestions: List[str],
        confidence: ConfidenceLevel,
    ) -> str:
        """Build a validated response with appropriate caveats."""
        
        # If high confidence, return original
        if confidence == ConfidenceLevel.HIGH and not issues:
            return original
        
        # Build response with validation info
        parts = [original]
        
        # Add confidence indicator
        conf_emoji = {
            ConfidenceLevel.HIGH: "✅",
            ConfidenceLevel.MEDIUM: "⚠️",
            ConfidenceLevel.LOW: "❓",
            ConfidenceLevel.UNKNOWN: "❔",
        }
        
        if confidence != ConfidenceLevel.HIGH:
            parts.append(f"\n\n---\n{conf_emoji[confidence]} **Validation Notes:**")
            
            if issues:
                for issue in issues:
                    parts.append(f"\n- {issue}")
            
            if suggestions:
                parts.append("\n\n**Suggestions:**")
                for sug in suggestions:
                    parts.append(f"\n- {sug}")
            
            # Add metadata if available
            if metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items() 
                                    if k not in ["last_updated"] and v)
                if meta_str:
                    parts.append(f"\n\n**Known metadata:** {meta_str}")
        
        return "".join(parts)
    
    async def validate_with_llm(
        self,
        tool_result: str,
        user_query: str,
    ) -> ValidationResult:
        """
        Use LLM to validate if tool result answers user query.
        Only used for complex cases or when confidence is low.
        """
        if not self.llm_client:
            return ValidationResult(
                is_valid=True,
                confidence=ConfidenceLevel.UNKNOWN,
                original_response=tool_result,
                validated_response=tool_result,
                issues=["LLM validation not available"],
            )
        
        validation_prompt = f"""You are a validation assistant. Check if the tool result properly answers the user's query.

User Query: {user_query}

Tool Result:
{tool_result}

Analyze:
1. Does the result actually answer what the user asked?
2. Are there any misleading claims (e.g., claiming data matches criteria when it can't be verified)?
3. Is important context missing?

Respond with JSON:
{{
    "is_valid": true/false,
    "confidence": "high/medium/low",
    "issues": ["list of issues"],
    "corrected_response": "the tool result with corrections/caveats added, or null if no changes needed"
}}
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": validation_prompt}],
                response_format={"type": "json_object"},
                max_tokens=1000,
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return ValidationResult(
                is_valid=result.get("is_valid", True),
                confidence=ConfidenceLevel(result.get("confidence", "medium")),
                original_response=tool_result,
                validated_response=result.get("corrected_response") or tool_result,
                issues=result.get("issues", []),
            )
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return ValidationResult(
                is_valid=True,
                confidence=ConfidenceLevel.UNKNOWN,
                original_response=tool_result,
                validated_response=tool_result,
                issues=[f"Validation error: {e}"],
            )


# Global validator instance
_validator: Optional[ResponseValidator] = None


def get_validator(llm_client=None, model: str = "gpt-4o") -> ResponseValidator:
    """Get or create the global validator instance."""
    global _validator
    if _validator is None:
        _validator = ResponseValidator(llm_client, model)
    elif llm_client and _validator.llm_client is None:
        _validator.llm_client = llm_client
        _validator.model = model
    return _validator


def validate_tool_response(
    tool_name: str,
    tool_result: Dict[str, Any],
    response_message: str,
    user_message: str,
    llm_client=None,
) -> ValidationResult:
    """
    Convenience function to validate a tool response.
    Uses the global validator to maintain conversation context.
    
    Args:
        tool_name: Name of the tool that was executed
        tool_result: The data returned by the tool
        response_message: The formatted message to show user
        user_message: The original user message
        llm_client: Optional LLM client for complex validation
    
    Returns:
        ValidationResult with validated response
    """
    validator = get_validator(llm_client)
    
    # Update context with user message (validator tracks conversation)
    validator.context.add_message("user", user_message)
    
    # Route to appropriate validator
    if tool_name == "scan_data":
        return validator.validate_scan_result(tool_result, response_message)
    elif tool_name == "search_databases":
        return validator.validate_search_result(tool_result, response_message)
    else:
        # Default: return as-is with medium confidence
        return ValidationResult(
            is_valid=True,
            confidence=ConfidenceLevel.MEDIUM,
            original_response=response_message,
            validated_response=response_message,
        )
