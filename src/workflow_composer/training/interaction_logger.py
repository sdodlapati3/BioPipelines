"""
Interaction Logger
==================

Logs real user interactions for training data collection.
Captures queries, intents, tool selections, workflows, and feedback.
"""

import json
import logging
import sqlite3
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum
import re

from .config import LoggerConfig

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of logged interactions."""
    QUERY = "query"
    INTENT_PARSE = "intent_parse"
    TOOL_SELECTION = "tool_selection"
    WORKFLOW_GENERATION = "workflow_generation"
    FEEDBACK = "feedback"
    EXECUTION = "execution"


class FeedbackType(Enum):
    """Types of user feedback."""
    ACCEPT = "accept"
    ACCEPT_WITH_MODIFICATIONS = "accept_modified"
    REJECT = "reject"
    RATING = "rating"


@dataclass
class Interaction:
    """A complete user interaction record."""
    
    id: str
    session_id: str
    timestamp: str
    
    # Core data
    query: str
    interaction_type: InteractionType
    
    # Intent data
    intent: Optional[Dict[str, Any]] = None
    intent_confidence: float = 0.0
    
    # Tool data
    tools_selected: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    
    # Workflow data
    workflow_generated: Optional[str] = None
    workflow_valid: bool = False
    
    # User feedback
    feedback_type: Optional[FeedbackType] = None
    feedback_text: Optional[str] = None
    modifications: Optional[str] = None
    rating: Optional[int] = None
    
    # Execution data
    execution_success: Optional[bool] = None
    execution_error: Optional[str] = None
    
    # Quality metrics
    quality_score: float = 0.5
    
    # User info (anonymized)
    user_hash: str = "anonymous"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['interaction_type'] = self.interaction_type.value
        if self.feedback_type:
            data['feedback_type'] = self.feedback_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Interaction':
        """Create from dictionary."""
        data = data.copy()
        data['interaction_type'] = InteractionType(data['interaction_type'])
        if data.get('feedback_type'):
            data['feedback_type'] = FeedbackType(data['feedback_type'])
        return cls(**data)
    
    def to_training_example(self, system_prompt: str = "") -> Dict[str, Any]:
        """Convert to training example format."""
        
        # Build response based on interaction type
        response_parts = []
        
        if self.intent:
            response_parts.append("## Analysis Intent")
            for key, value in self.intent.items():
                if value and key not in ['raw', 'confidence']:
                    response_parts.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        
        if self.tools_selected:
            response_parts.append("\n## Selected Tools")
            for tool in self.tools_selected:
                response_parts.append(f"- {tool}")
        
        if self.workflow_generated:
            response_parts.append("\n## Generated Workflow")
            response_parts.append(f"```nextflow\n{self.workflow_generated}\n```")
        
        messages = [
            {"role": "user", "content": self.query},
            {"role": "assistant", "content": "\n".join(response_parts)},
        ]
        
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        return {
            "id": self.id,
            "messages": messages,
            "metadata": {
                "source": "interaction_log",
                "session_id": self.session_id,
                "quality_score": self.quality_score,
                "feedback": self.feedback_type.value if self.feedback_type else None,
                "execution_success": self.execution_success,
            }
        }


class InteractionLogger:
    """Log and manage user interactions for training data."""
    
    def __init__(self, config: LoggerConfig = None):
        self.config = config or LoggerConfig()
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.config.storage_dir / self.config.db_file
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for interactions."""
        
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    timestamp TEXT,
                    query TEXT,
                    interaction_type TEXT,
                    intent TEXT,
                    intent_confidence REAL,
                    tools_selected TEXT,
                    tools_used TEXT,
                    workflow_generated TEXT,
                    workflow_valid INTEGER,
                    feedback_type TEXT,
                    feedback_text TEXT,
                    modifications TEXT,
                    rating INTEGER,
                    execution_success INTEGER,
                    execution_error TEXT,
                    quality_score REAL,
                    user_hash TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session ON interactions(session_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality ON interactions(quality_score)
            """)
            
            conn.commit()
    
    def _generate_id(self) -> str:
        """Generate unique interaction ID."""
        timestamp = datetime.now().isoformat()
        hash_val = hashlib.md5(f"{timestamp}{id(self)}".encode()).hexdigest()[:12]
        return f"int_{hash_val}"
    
    def _anonymize_user(self, user_id: str) -> str:
        """Anonymize user ID."""
        if self.config.anonymize and user_id:
            return hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return "anonymous"
    
    def _remove_pii(self, text: str) -> str:
        """Remove personally identifiable information."""
        if not self.config.remove_pii or not text:
            return text
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Remove paths that might contain usernames
        text = re.sub(r'/home/[^/\s]+', '/home/[USER]', text)
        text = re.sub(r'/Users/[^/\s]+', '/Users/[USER]', text)
        
        return text
    
    def log_query(
        self,
        session_id: str,
        query: str,
        user_id: str = None
    ) -> str:
        """Log a new query and return interaction ID."""
        
        interaction = Interaction(
            id=self._generate_id(),
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            query=self._remove_pii(query),
            interaction_type=InteractionType.QUERY,
            user_hash=self._anonymize_user(user_id),
        )
        
        self._save_interaction(interaction)
        return interaction.id
    
    def log_intent(
        self,
        interaction_id: str,
        intent: Dict[str, Any],
        confidence: float = 0.0
    ):
        """Log parsed intent for an interaction."""
        
        self._update_interaction(
            interaction_id,
            intent=json.dumps(intent),
            intent_confidence=confidence,
            interaction_type=InteractionType.INTENT_PARSE.value,
        )
    
    def log_tools(
        self,
        interaction_id: str,
        tools_selected: List[str],
        tools_used: List[str] = None
    ):
        """Log tool selection for an interaction."""
        
        self._update_interaction(
            interaction_id,
            tools_selected=json.dumps(tools_selected),
            tools_used=json.dumps(tools_used or []),
            interaction_type=InteractionType.TOOL_SELECTION.value,
        )
    
    def log_workflow(
        self,
        interaction_id: str,
        workflow: str,
        valid: bool = False
    ):
        """Log generated workflow for an interaction."""
        
        self._update_interaction(
            interaction_id,
            workflow_generated=self._remove_pii(workflow),
            workflow_valid=1 if valid else 0,
            interaction_type=InteractionType.WORKFLOW_GENERATION.value,
        )
    
    def log_feedback(
        self,
        interaction_id: str,
        feedback_type: str,
        feedback_text: str = None,
        modifications: str = None,
        rating: int = None
    ):
        """Log user feedback for an interaction.
        
        Args:
            interaction_id: The ID of the interaction (or session_id to find latest)
            feedback_type: Type of feedback - 'accept', 'accept_modified', 'reject', 'rating'
            feedback_text: Optional text feedback
            modifications: Optional modification details
            rating: Optional numeric rating
        """
        # Convert string to enum if needed
        if isinstance(feedback_type, str):
            try:
                feedback_type_enum = FeedbackType(feedback_type)
            except ValueError:
                # Try to find by name
                feedback_type_enum = FeedbackType.RATING
        else:
            feedback_type_enum = feedback_type
        
        self._update_interaction(
            interaction_id,
            feedback_type=feedback_type_enum.value,
            feedback_text=self._remove_pii(feedback_text) if feedback_text else None,
            modifications=self._remove_pii(modifications) if modifications else None,
            rating=rating,
            interaction_type=InteractionType.FEEDBACK.value,
        )
        
        # Recalculate quality score with feedback
        self._update_quality_score(interaction_id)
    
    def log_execution(
        self,
        interaction_id: str,
        success: bool,
        error: str = None
    ):
        """Log execution result for an interaction."""
        
        self._update_interaction(
            interaction_id,
            execution_success=1 if success else 0,
            execution_error=error,
            interaction_type=InteractionType.EXECUTION.value,
        )
        
        # Recalculate quality score with execution result
        self._update_quality_score(interaction_id)
    
    def _save_interaction(self, interaction: Interaction):
        """Save interaction to database."""
        
        with sqlite3.connect(self._db_path) as conn:
            data = interaction.to_dict()
            
            # Convert all non-primitive types to JSON strings
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = json.dumps(value)
                elif isinstance(value, Enum):
                    data[key] = value.value
            
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            
            conn.execute(
                f"INSERT INTO interactions ({columns}) VALUES ({placeholders})",
                list(data.values())
            )
            conn.commit()
    
    def _update_interaction(self, interaction_id: str, **kwargs):
        """Update interaction fields."""
        
        with sqlite3.connect(self._db_path) as conn:
            updates = ', '.join([f"{k} = ?" for k in kwargs.keys()])
            conn.execute(
                f"UPDATE interactions SET {updates} WHERE id = ?",
                list(kwargs.values()) + [interaction_id]
            )
            conn.commit()
    
    def _update_quality_score(self, interaction_id: str):
        """Recalculate quality score for an interaction."""
        
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM interactions WHERE id = ?",
                (interaction_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return
            
            # Calculate quality score based on available signals
            score = 0.5  # Base score
            
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            
            # High confidence intent
            if data.get('intent_confidence', 0) > 0.8:
                score += 0.15
            
            # Valid workflow
            if data.get('workflow_valid'):
                score += 0.1
            
            # Positive feedback
            feedback = data.get('feedback_type')
            if feedback == 'accept':
                score += 0.2
            elif feedback == 'accept_modified':
                score += 0.1
            elif feedback == 'reject':
                score -= 0.2
            
            # High rating
            rating = data.get('rating')
            if rating:
                score += (rating - 3) * 0.05  # -0.1 to +0.1
            
            # Successful execution
            if data.get('execution_success'):
                score += 0.1
            elif data.get('execution_success') is not None:
                score -= 0.1
            
            score = max(0.0, min(1.0, score))
            
            conn.execute(
                "UPDATE interactions SET quality_score = ? WHERE id = ?",
                (score, interaction_id)
            )
            conn.commit()
    
    def get_interaction(self, interaction_id: str) -> Optional[Interaction]:
        """Get a single interaction by ID."""
        
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM interactions WHERE id = ?",
                (interaction_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_interaction(dict(row))
    
    def get_session_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all interactions for a session as dictionaries."""
        
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM interactions WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            
            interactions = []
            for row in cursor:
                row_dict = dict(row)
                # Parse JSON fields
                if row_dict.get('intent'):
                    row_dict['intent'] = json.loads(row_dict['intent'])
                if row_dict.get('tools_selected'):
                    row_dict['tools_selected'] = json.loads(row_dict['tools_selected'])
                if row_dict.get('tools_used'):
                    row_dict['tools_used'] = json.loads(row_dict['tools_used'])
                interactions.append(row_dict)
            
            return interactions
    
    def get_high_quality_interactions(
        self,
        min_score: float = None,
        limit: int = None
    ) -> List[Interaction]:
        """Get interactions above quality threshold."""
        
        min_score = min_score or self.config.min_quality_score
        
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM interactions WHERE quality_score >= ?"
            params = [min_score]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = conn.execute(query, params)
            
            return [self._row_to_interaction(dict(row)) for row in cursor]
    
    def _row_to_interaction(self, row: Dict) -> Interaction:
        """Convert database row to Interaction object."""
        
        # Parse JSON fields
        if row.get('intent'):
            row['intent'] = json.loads(row['intent'])
        if row.get('tools_selected'):
            row['tools_selected'] = json.loads(row['tools_selected'])
        if row.get('tools_used'):
            row['tools_used'] = json.loads(row['tools_used'])
        
        # Convert enums
        row['interaction_type'] = InteractionType(row['interaction_type'])
        if row.get('feedback_type'):
            row['feedback_type'] = FeedbackType(row['feedback_type'])
        
        # Convert booleans
        if row.get('workflow_valid') is not None:
            row['workflow_valid'] = bool(row['workflow_valid'])
        if row.get('execution_success') is not None:
            row['execution_success'] = bool(row['execution_success'])
        
        return Interaction(**row)
    
    def export_training_data(
        self,
        output_path: Path,
        min_quality: float = None,
        system_prompt: str = ""
    ) -> int:
        """Export high-quality interactions as training data."""
        
        interactions = self.get_high_quality_interactions(min_quality)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with open(output_path, 'w') as f:
            for interaction in interactions:
                # Only export complete interactions
                if interaction.intent and interaction.workflow_generated:
                    example = interaction.to_training_example(system_prompt)
                    f.write(json.dumps(example) + '\n')
                    count += 1
        
        logger.info(f"Exported {count} training examples to {output_path}")
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        
        with sqlite3.connect(self._db_path) as conn:
            # Total count
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            
            # Quality distribution
            high_quality = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE quality_score >= ?",
                (self.config.min_quality_score,)
            ).fetchone()[0]
            
            # By type
            type_counts = {}
            for row in conn.execute(
                "SELECT interaction_type, COUNT(*) FROM interactions GROUP BY interaction_type"
            ):
                type_counts[row[0]] = row[1]
            
            # Average quality
            avg_quality = conn.execute(
                "SELECT AVG(quality_score) FROM interactions"
            ).fetchone()[0] or 0.0
            
            return {
                "total_interactions": total,
                "high_quality_count": high_quality,
                "high_quality_ratio": high_quality / total if total > 0 else 0,
                "by_type": type_counts,
                "average_quality": round(avg_quality, 3),
            }


# Global logger instance
_logger_instance = None


def get_interaction_logger(config: LoggerConfig = None) -> InteractionLogger:
    """Get or create the global interaction logger."""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = InteractionLogger(config)
    
    return _logger_instance
