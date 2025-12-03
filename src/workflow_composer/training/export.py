"""
Training Data Export
====================

Export training data in various formats for different fine-tuning frameworks.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ExportConfig

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Result of an export operation."""
    
    format: str
    output_path: Path
    example_count: int
    file_size_bytes: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "output_path": str(self.output_path),
            "example_count": self.example_count,
            "file_size_bytes": self.file_size_bytes,
            "timestamp": self.timestamp,
        }


class BaseExporter(ABC):
    """Base class for training data exporters."""
    
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
    
    @abstractmethod
    def export(
        self, 
        examples: List[Dict[str, Any]], 
        output_path: Path
    ) -> ExportResult:
        """Export examples to the specified format."""
        pass
    
    def _format_system_prompt(self) -> str:
        """Get the system prompt for chat format."""
        return self.config.system_prompt or """You are a bioinformatics workflow assistant. You help users design and execute computational biology pipelines using Nextflow. When given a research question or analysis task, identify the appropriate tools and create workflow configurations."""


class OpenAIChatExporter(BaseExporter):
    """Export in OpenAI chat format for fine-tuning."""
    
    def export(
        self, 
        examples: List[Dict[str, Any]], 
        output_path: Path
    ) -> ExportResult:
        """Export to OpenAI chat JSONL format."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in examples:
                chat_example = self._convert_example(example)
                f.write(json.dumps(chat_example) + '\n')
        
        file_size = output_path.stat().st_size
        
        return ExportResult(
            format="openai_chat",
            output_path=output_path,
            example_count=len(examples),
            file_size_bytes=file_size,
        )
    
    def _convert_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an example to OpenAI chat format."""
        
        messages = [
            {
                "role": "system",
                "content": self._format_system_prompt()
            },
            {
                "role": "user", 
                "content": example.get('query', '')
            },
            {
                "role": "assistant",
                "content": self._format_response(example)
            }
        ]
        
        return {"messages": messages}
    
    def _format_response(self, example: Dict[str, Any]) -> str:
        """Format the assistant response."""
        
        parts = []
        
        # Intent interpretation
        intent = example.get('intent', {})
        if intent:
            analysis_type = intent.get('analysis_type', 'analysis')
            parts.append(f"I'll help you set up a {analysis_type} workflow.")
        
        # Tools selection
        tools = example.get('tools', [])
        if tools:
            tools_str = ", ".join(tools)
            parts.append(f"I recommend using the following tools: {tools_str}.")
        
        # Explanation
        if example.get('explanation'):
            parts.append(example['explanation'])
        
        # Workflow
        if example.get('workflow'):
            parts.append(f"\nHere's the workflow configuration:\n\n```nextflow\n{example['workflow']}\n```")
        
        return "\n\n".join(parts) if parts else "I understand your request. Let me help you with that."


class AlpacaExporter(BaseExporter):
    """Export in Alpaca instruction format."""
    
    def export(
        self, 
        examples: List[Dict[str, Any]], 
        output_path: Path
    ) -> ExportResult:
        """Export to Alpaca JSON format."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        alpaca_examples = [self._convert_example(ex) for ex in examples]
        
        with open(output_path, 'w') as f:
            json.dump(alpaca_examples, f, indent=2)
        
        file_size = output_path.stat().st_size
        
        return ExportResult(
            format="alpaca",
            output_path=output_path,
            example_count=len(examples),
            file_size_bytes=file_size,
        )
    
    def _convert_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Convert to Alpaca format."""
        
        return {
            "instruction": example.get('query', ''),
            "input": self._format_input(example),
            "output": self._format_output(example),
        }
    
    def _format_input(self, example: Dict[str, Any]) -> str:
        """Format optional input context."""
        
        parts = []
        
        if example.get('organism'):
            parts.append(f"Organism: {example['organism']}")
        
        if example.get('data_type'):
            parts.append(f"Data type: {example['data_type']}")
        
        if example.get('parameters'):
            params = example['parameters']
            if isinstance(params, dict):
                for k, v in params.items():
                    parts.append(f"{k}: {v}")
        
        return "\n".join(parts)
    
    def _format_output(self, example: Dict[str, Any]) -> str:
        """Format the expected output."""
        
        parts = []
        
        # Analysis type
        intent = example.get('intent', {})
        if intent and intent.get('analysis_type'):
            parts.append(f"Analysis: {intent['analysis_type']}")
        
        # Tools
        tools = example.get('tools', [])
        if tools:
            parts.append(f"Tools: {', '.join(tools)}")
        
        # Workflow
        if example.get('workflow'):
            parts.append(f"\n{example['workflow']}")
        
        return "\n".join(parts)


class ShareGPTExporter(BaseExporter):
    """Export in ShareGPT conversation format."""
    
    def export(
        self, 
        examples: List[Dict[str, Any]], 
        output_path: Path
    ) -> ExportResult:
        """Export to ShareGPT JSON format."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        conversations = [self._convert_example(ex) for ex in examples]
        
        with open(output_path, 'w') as f:
            json.dump(conversations, f, indent=2)
        
        file_size = output_path.stat().st_size
        
        return ExportResult(
            format="sharegpt",
            output_path=output_path,
            example_count=len(examples),
            file_size_bytes=file_size,
        )
    
    def _convert_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to ShareGPT format."""
        
        conversations = [
            {"from": "human", "value": example.get('query', '')},
            {"from": "gpt", "value": self._format_response(example)},
        ]
        
        return {"conversations": conversations}
    
    def _format_response(self, example: Dict[str, Any]) -> str:
        """Format the GPT response."""
        
        # Similar to OpenAI format
        parts = []
        
        intent = example.get('intent', {})
        if intent and intent.get('analysis_type'):
            parts.append(f"This request is for {intent['analysis_type']} analysis.")
        
        tools = example.get('tools', [])
        if tools:
            parts.append(f"Recommended tools: {', '.join(tools)}")
        
        if example.get('explanation'):
            parts.append(example['explanation'])
        
        if example.get('workflow'):
            parts.append(f"```nextflow\n{example['workflow']}\n```")
        
        return "\n\n".join(parts) if parts else example.get('response', '')


class AxolotlExporter(BaseExporter):
    """Export for Axolotl fine-tuning framework."""
    
    def export(
        self, 
        examples: List[Dict[str, Any]], 
        output_path: Path
    ) -> ExportResult:
        """Export to Axolotl-compatible format."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in examples:
                axolotl_example = self._convert_example(example)
                f.write(json.dumps(axolotl_example) + '\n')
        
        file_size = output_path.stat().st_size
        
        return ExportResult(
            format="axolotl",
            output_path=output_path,
            example_count=len(examples),
            file_size_bytes=file_size,
        )
    
    def _convert_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Convert to Axolotl format with tags."""
        
        prompt = f"### Instruction:\n{example.get('query', '')}"
        
        # Add input if present
        input_text = self._format_input(example)
        if input_text:
            prompt += f"\n\n### Input:\n{input_text}"
        
        response = self._format_response(example)
        
        return {
            "text": f"{prompt}\n\n### Response:\n{response}"
        }
    
    def _format_input(self, example: Dict[str, Any]) -> str:
        """Format input context."""
        
        parts = []
        
        if example.get('organism'):
            parts.append(f"Organism: {example['organism']}")
        
        if example.get('data_type'):
            parts.append(f"Data: {example['data_type']}")
        
        return "\n".join(parts)
    
    def _format_response(self, example: Dict[str, Any]) -> str:
        """Format response."""
        
        parts = []
        
        intent = example.get('intent', {})
        tools = example.get('tools', [])
        
        if intent.get('analysis_type'):
            parts.append(f"Analysis: {intent['analysis_type']}")
        
        if tools:
            parts.append(f"Tools: {', '.join(tools)}")
        
        if example.get('workflow'):
            parts.append(f"\n{example['workflow']}")
        
        return "\n".join(parts)


class IntentClassificationExporter(BaseExporter):
    """Export for intent classification task only."""
    
    def export(
        self, 
        examples: List[Dict[str, Any]], 
        output_path: Path
    ) -> ExportResult:
        """Export query-intent pairs for classification training."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in examples:
                intent = example.get('intent', {})
                if intent:
                    classification_example = {
                        "text": example.get('query', ''),
                        "label": intent.get('analysis_type', 'unknown'),
                        "confidence": intent.get('confidence', 0.5),
                    }
                    f.write(json.dumps(classification_example) + '\n')
        
        file_size = output_path.stat().st_size
        
        # Count only examples with intents
        valid_count = sum(1 for ex in examples if ex.get('intent'))
        
        return ExportResult(
            format="intent_classification",
            output_path=output_path,
            example_count=valid_count,
            file_size_bytes=file_size,
        )


class ToolSelectionExporter(BaseExporter):
    """Export for tool selection task only."""
    
    def export(
        self, 
        examples: List[Dict[str, Any]], 
        output_path: Path
    ) -> ExportResult:
        """Export query-tools pairs for tool selection training."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        valid_count = 0
        
        with open(output_path, 'w') as f:
            for example in examples:
                tools = example.get('tools', [])
                if tools:
                    tool_example = {
                        "query": example.get('query', ''),
                        "analysis_type": example.get('intent', {}).get('analysis_type'),
                        "tools": tools,
                    }
                    f.write(json.dumps(tool_example) + '\n')
                    valid_count += 1
        
        file_size = output_path.stat().st_size if output_path.exists() else 0
        
        return ExportResult(
            format="tool_selection",
            output_path=output_path,
            example_count=valid_count,
            file_size_bytes=file_size,
        )


class TrainingDataExporter:
    """Main exporter class that coordinates all export formats."""
    
    EXPORTERS = {
        "openai_chat": OpenAIChatExporter,
        "alpaca": AlpacaExporter,
        "sharegpt": ShareGPTExporter,
        "axolotl": AxolotlExporter,
        "intent": IntentClassificationExporter,
        "tools": ToolSelectionExporter,
    }
    
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
    
    def export(
        self,
        examples: List[Dict[str, Any]],
        format: str,
        output_path: Optional[Path] = None,
    ) -> ExportResult:
        """Export examples in the specified format."""
        
        if format not in self.EXPORTERS:
            raise ValueError(f"Unknown format: {format}. Available: {list(self.EXPORTERS.keys())}")
        
        exporter = self.EXPORTERS[format](self.config)
        
        if output_path is None:
            output_path = self.config.output_dir / f"training_{format}.jsonl"
        
        return exporter.export(examples, output_path)
    
    def export_all_formats(
        self,
        examples: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, ExportResult]:
        """Export to all available formats."""
        
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for format_name in self.EXPORTERS:
            suffix = ".jsonl" if format_name not in ["alpaca", "sharegpt"] else ".json"
            output_path = output_dir / f"training_{format_name}{suffix}"
            
            try:
                result = self.export(examples, format_name, output_path)
                results[format_name] = result
                logger.info(f"Exported {result.example_count} examples to {format_name}")
            except Exception as e:
                logger.error(f"Failed to export {format_name}: {e}")
        
        return results
    
    def load_and_export(
        self,
        input_path: Path,
        format: str,
        output_path: Optional[Path] = None,
    ) -> ExportResult:
        """Load from JSONL and export to specified format."""
        
        examples = []
        
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        
        return self.export(examples, format, output_path)


def export_training_data(
    examples: List[Dict[str, Any]],
    format: str = "openai_chat",
    output_path: Optional[Path] = None,
) -> ExportResult:
    """Convenience function to export training data."""
    
    exporter = TrainingDataExporter()
    return exporter.export(examples, format, output_path)


def export_all_formats(
    examples: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, ExportResult]:
    """Export to all formats."""
    
    exporter = TrainingDataExporter()
    return exporter.export_all_formats(examples, output_dir)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    sample_examples = [
        {
            "id": "example_1",
            "query": "I want to analyze RNA-seq data from mouse liver samples",
            "intent": {
                "analysis_type": "rna_seq",
                "confidence": 0.95,
            },
            "tools": ["fastp", "star", "featurecounts", "deseq2"],
            "workflow": 'include { FASTP } from "./modules/fastp"\nworkflow { FASTP(reads) }',
            "explanation": "This workflow performs RNA-seq analysis using STAR for alignment and DESeq2 for differential expression.",
        }
    ]
    
    exporter = TrainingDataExporter()
    results = exporter.export_all_formats(sample_examples, Path("/tmp/training_export"))
    
    for format_name, result in results.items():
        print(f"{format_name}: {result.example_count} examples -> {result.output_path}")
