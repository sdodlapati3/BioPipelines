"""
Experiment Runner
==================

Orchestrates evaluation experiments:
- Runs parser on generated conversations
- Stores results in database
- Generates reports
- Tracks progress over time
- Supports A/B testing of configurations
"""

import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .database import get_database, EvaluationDatabase
    from .conversation_generator import ConversationGenerator, populate_database
    from .enhanced_metrics import (
        EnhancedEvaluator, 
        ConversationEvaluation,
        TurnEvaluation,
    )
except ImportError:
    from database import get_database, EvaluationDatabase
    from conversation_generator import ConversationGenerator, populate_database
    from enhanced_metrics import (
        EnhancedEvaluator, 
        ConversationEvaluation,
        TurnEvaluation,
    )

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an evaluation experiment."""
    name: str
    description: str
    parser_type: str = "unified_ensemble"  # unified_ensemble, rule_only, llm_only
    parser_weights: Dict[str, float] = None
    use_llm_metrics: bool = False
    conversation_categories: List[str] = None
    max_conversations: int = 100
    parallel_workers: int = 4
    
    def __post_init__(self):
        if self.parser_weights is None:
            self.parser_weights = {
                'rule': 0.25,
                'semantic': 0.30,
                'ner': 0.20,
                'llm': 0.15,
                'rag': 0.10,
            }
        if self.conversation_categories is None:
            self.conversation_categories = []


class ExperimentRunner:
    """
    Runs evaluation experiments and stores results.
    
    Features:
    - Run evaluations on database conversations
    - Store results for trend tracking
    - Generate detailed reports
    - Support for parallel evaluation
    - A/B testing support
    """
    
    def __init__(
        self,
        db: EvaluationDatabase = None,
        parser_func: Callable = None,
    ):
        """
        Initialize runner.
        
        Args:
            db: Database instance (uses default if None)
            parser_func: Parser function (loads default if None)
        """
        self.db = db or get_database()
        self._parser_func = parser_func
        self._parser = None
    
    def _get_parser(self):
        """Lazy load the parser."""
        if self._parser_func is not None:
            return self._parser_func
        
        if self._parser is not None:
            return lambda q: self._parse_with_ensemble(q)
        
        try:
            from workflow_composer.agents.intent.unified_ensemble import UnifiedEnsembleParser
            self._parser = UnifiedEnsembleParser()
            return lambda q: self._parse_with_ensemble(q)
        except ImportError:
            logger.warning("Could not import UnifiedEnsembleParser, using mock")
            return self._mock_parser
    
    def _parse_with_ensemble(self, query: str) -> Dict[str, Any]:
        """Parse using the ensemble parser."""
        result = self._parser.parse(query)
        # result.intent is already a string, not an Enum
        intent = result.intent if isinstance(result.intent, str) else (result.intent.name if result.intent else 'UNKNOWN')
        
        # Convert entities from list of BioEntity to dict for metrics compatibility
        # Use ONLY BioEntity list (canonical uppercase types) - slots duplicate these
        entities_dict = {}
        if result.entities:
            for entity in result.entities:
                entity_type = entity.entity_type if hasattr(entity, 'entity_type') else 'UNKNOWN'
                # Ensure uppercase for consistency with expected format
                entity_type = entity_type.upper()
                
                # Get the canonical value (preferred) or original text
                entity_value = getattr(entity, 'canonical', None) or (entity.text if hasattr(entity, 'text') else str(entity))
                
                # Store as single value, not list (expected format uses single values)
                # If we already have this type, keep first occurrence
                if entity_type not in entities_dict:
                    entities_dict[entity_type] = entity_value
        
        # Map intent to a suggested tool
        intent_to_tool = {
            # Data operations
            'DATA_SEARCH': 'search_databases',
            'DATA_DISCOVERY': 'search_databases',
            'DATA_DOWNLOAD': 'download_data',
            'DATA_SCAN': 'scan_data',
            'DATA_DESCRIBE': 'describe_data',
            # Workflow operations
            'WORKFLOW_CREATE': 'generate_workflow',
            'WORKFLOW_GENERATION': 'generate_workflow',
            'WORKFLOW_VISUALIZE': 'visualize_workflow',
            # Job operations
            'JOB_SUBMIT': 'submit_job',
            'RUN_WORKFLOW': 'execute_workflow',
            'WORKFLOW_EXECUTE': 'execute_workflow',
            'JOB_STATUS': 'check_job_status',
            'JOB_LIST': 'list_jobs',
            'JOB_LOGS': 'show_logs',
            'JOB_CANCEL': 'cancel_job',
            'DIAGNOSE_ERROR': 'diagnose_error',
            # Education - CRITICAL for education pass rate
            'EDUCATION_HELP': 'show_help',
            'EDUCATION_EXPLAIN': 'explain_concept',
            'HELP': 'show_help',
            'EDUCATION': 'show_help',
            'META_GREETING': 'show_help',
            'GREETING': 'show_help',
            # References
            'REFERENCE_CHECK': 'check_reference',
            'REFERENCE_DOWNLOAD': 'download_reference',
            # Analysis
            'ANALYSIS_INTERPRET': 'interpret_results',
            # Meta operations
            'META_CONFIRM': 'confirm_action',
            'META_CANCEL': 'cancel_action',
            'META_UNKNOWN': 'clarify_intent',
        }
        tool = intent_to_tool.get(intent, 'search_databases')
        
        return {
            'intent': intent,
            'entities': entities_dict,
            'tool': tool,
            'confidence': result.confidence,
        }
    
    def _mock_parser(self, query: str) -> Dict[str, Any]:
        """Mock parser for testing."""
        return {
            'intent': 'DATA_SEARCH',
            'entities': {},
            'tool': 'search_databases',
            'confidence': 0.5,
        }
    
    def ensure_conversations(
        self,
        target_count: int = 1000,
    ) -> int:
        """Ensure database has enough conversations."""
        current = self.db.get_conversation_count()
        
        if current >= target_count:
            logger.info(f"Database already has {current} conversations")
            return current
        
        logger.info(f"Generating {target_count - current} more conversations...")
        added = populate_database(target_count)
        
        return self.db.get_conversation_count()
    
    def run_experiment(
        self,
        config: ExperimentConfig,
    ) -> str:
        """
        Run an evaluation experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        logger.info(f"Starting experiment: {config.name}")
        
        # Create experiment in database
        experiment_id = self.db.create_experiment(
            name=config.name,
            description=config.description,
            parser_config={
                'type': config.parser_type,
                'weights': config.parser_weights,
            },
        )
        
        try:
            # Get conversations to evaluate
            conversations = []
            if config.conversation_categories:
                for category in config.conversation_categories:
                    convs = self.db.get_conversations(
                        category=category,
                        limit=config.max_conversations // len(config.conversation_categories),
                    )
                    conversations.extend(convs)
            else:
                conversations = self.db.get_conversations(
                    limit=config.max_conversations,
                )
            
            logger.info(f"Evaluating {len(conversations)} conversations...")
            
            # Create evaluator
            evaluator = EnhancedEvaluator(
                use_llm=config.use_llm_metrics,
                use_semantic=False,
            )
            
            # Get parser
            parser = self._get_parser()
            
            # Run evaluations
            results = []
            start_time = time.time()
            
            for i, conv in enumerate(conversations):
                try:
                    eval_result = evaluator.evaluate_conversation(
                        conversation_id=conv.id,
                        conversation_name=conv.name,
                        category=conv.category,
                        turns=conv.turns,
                        parser_func=parser,
                    )
                    results.append(eval_result)
                    
                    # Store individual result
                    self.db.add_result(
                        experiment_id=experiment_id,
                        conversation_id=conv.id,
                        intent_accuracy=eval_result.intent_accuracy,
                        entity_f1=eval_result.entity_f1,
                        tool_accuracy=eval_result.tool_accuracy,
                        latency_ms=eval_result.avg_latency_ms,
                        passed=eval_result.passed,
                        turns_detail=[
                            {
                                'query': t.query,
                                'expected_intent': t.expected_intent,
                                'predicted_intent': t.predicted_intent,
                                'intent_score': t.intent_score.score,
                                'entity_score': t.entity_score.score,
                            }
                            for t in eval_result.turns
                        ],
                        parser_config=asdict(config),
                    )
                    
                    if (i + 1) % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed
                        logger.info(
                            f"Progress: {i + 1}/{len(conversations)} "
                            f"({rate:.1f} conv/sec)"
                        )
                        
                except Exception as e:
                    logger.error(f"Error evaluating {conv.id}: {e}")
                    traceback.print_exc()
            
            # Generate report
            report = evaluator.generate_report(results)
            
            # Update experiment with final results
            self.db.update_experiment_results(
                experiment_id=experiment_id,
                total=len(results),
                passed=report['summary']['passed_conversations'],
                intent_acc=report['summary']['avg_intent_accuracy'],
                entity_f1=report['summary']['avg_entity_f1'],
                tool_acc=report['summary']['avg_tool_accuracy'],
                avg_latency=report['summary']['avg_latency_ms'],
                status='completed',
                notes=json.dumps(report['by_category']),
            )
            
            # Record metrics for trend tracking
            self.db.record_metric('pass_rate', report['summary']['pass_rate'], 
                                experiment_id=experiment_id)
            self.db.record_metric('intent_accuracy', report['summary']['avg_intent_accuracy'],
                                experiment_id=experiment_id)
            self.db.record_metric('entity_f1', report['summary']['avg_entity_f1'],
                                experiment_id=experiment_id)
            
            elapsed = time.time() - start_time
            logger.info(
                f"Experiment completed in {elapsed:.1f}s\n"
                f"  Pass Rate: {report['summary']['pass_rate']*100:.1f}%\n"
                f"  Intent Accuracy: {report['summary']['avg_intent_accuracy']*100:.1f}%\n"
                f"  Entity F1: {report['summary']['avg_entity_f1']*100:.1f}%\n"
                f"  Tool Accuracy: {report['summary']['avg_tool_accuracy']*100:.1f}%\n"
                f"  Avg Latency: {report['summary']['avg_latency_ms']:.0f}ms"
            )
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            traceback.print_exc()
            self.db.update_experiment_results(
                experiment_id=experiment_id,
                total=0, passed=0, intent_acc=0, entity_f1=0, tool_acc=0, avg_latency=0,
                status='failed',
                notes=str(e),
            )
            raise
    
    def compare_experiments(
        self,
        exp_id_1: str,
        exp_id_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two experiments.
        
        Returns:
            Comparison report with deltas
        """
        exp1 = self.db.get_experiment(exp_id_1)
        exp2 = self.db.get_experiment(exp_id_2)
        
        if not exp1 or not exp2:
            raise ValueError("One or both experiments not found")
        
        return {
            'experiment_1': {
                'id': exp1.id,
                'name': exp1.name,
                'pass_rate': exp1.passed_conversations / exp1.total_conversations if exp1.total_conversations else 0,
                'intent_accuracy': exp1.overall_intent_accuracy,
                'entity_f1': exp1.overall_entity_f1,
                'latency_ms': exp1.avg_latency_ms,
            },
            'experiment_2': {
                'id': exp2.id,
                'name': exp2.name,
                'pass_rate': exp2.passed_conversations / exp2.total_conversations if exp2.total_conversations else 0,
                'intent_accuracy': exp2.overall_intent_accuracy,
                'entity_f1': exp2.overall_entity_f1,
                'latency_ms': exp2.avg_latency_ms,
            },
            'deltas': {
                'pass_rate': (
                    (exp2.passed_conversations / exp2.total_conversations if exp2.total_conversations else 0) -
                    (exp1.passed_conversations / exp1.total_conversations if exp1.total_conversations else 0)
                ),
                'intent_accuracy': exp2.overall_intent_accuracy - exp1.overall_intent_accuracy,
                'entity_f1': exp2.overall_entity_f1 - exp1.overall_entity_f1,
                'latency_ms': exp2.avg_latency_ms - exp1.avg_latency_ms,
            }
        }
    
    def get_failure_analysis(
        self,
        experiment_id: str,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Get detailed failure analysis.
        
        Returns:
            List of failure details with patterns
        """
        failures = self.db.get_failed_results(experiment_id, limit)
        
        # Group failures by pattern
        patterns = {}
        for f in failures:
            turns = json.loads(f['turns_detail_json']) if f['turns_detail_json'] else []
            for turn in turns:
                if turn.get('intent_score', 1.0) < 1.0:
                    pattern_key = f"{turn.get('expected_intent')} -> {turn.get('predicted_intent')}"
                    if pattern_key not in patterns:
                        patterns[pattern_key] = {
                            'expected': turn.get('expected_intent'),
                            'predicted': turn.get('predicted_intent'),
                            'count': 0,
                            'examples': [],
                        }
                    patterns[pattern_key]['count'] += 1
                    if len(patterns[pattern_key]['examples']) < 5:
                        patterns[pattern_key]['examples'].append(turn.get('query', ''))
        
        # Sort by frequency
        sorted_patterns = sorted(
            patterns.values(),
            key=lambda x: x['count'],
            reverse=True
        )
        
        return sorted_patterns
    
    def generate_html_report(
        self,
        experiment_id: str,
        output_path: Path = None,
    ) -> str:
        """Generate HTML report for experiment."""
        exp = self.db.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        category_results = self.db.get_category_results(experiment_id)
        failures = self.get_failure_analysis(experiment_id, limit=20)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report: {exp.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .pass {{ color: #4CAF50; }}
        .fail {{ color: #f44336; }}
        h2 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>Evaluation Report: {exp.name}</h1>
    <p>{exp.description}</p>
    <p><strong>Completed:</strong> {exp.completed_at}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <div class="metric-value">{exp.passed_conversations}/{exp.total_conversations}</div>
            <div class="metric-label">Pass Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{exp.overall_intent_accuracy*100:.1f}%</div>
            <div class="metric-label">Intent Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{exp.overall_entity_f1*100:.1f}%</div>
            <div class="metric-label">Entity F1</div>
        </div>
        <div class="metric">
            <div class="metric-value">{exp.overall_tool_accuracy*100:.1f}%</div>
            <div class="metric-label">Tool Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{exp.avg_latency_ms:.0f}ms</div>
            <div class="metric-label">Avg Latency</div>
        </div>
    </div>
    
    <h2>Results by Category</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Total</th>
            <th>Passed</th>
            <th>Pass Rate</th>
            <th>Intent Acc</th>
            <th>Entity F1</th>
        </tr>
"""
        
        for cat, data in sorted(category_results.items()):
            pass_class = 'pass' if data['pass_rate'] >= 0.7 else 'fail'
            html += f"""        <tr>
            <td>{cat}</td>
            <td>{data['total']}</td>
            <td>{data['passed']}</td>
            <td class="{pass_class}">{data['pass_rate']*100:.1f}%</td>
            <td>{data['intent_accuracy']*100:.1f}%</td>
            <td>{data['entity_f1']*100:.1f}%</td>
        </tr>
"""
        
        html += """    </table>
    
    <h2>Top Failure Patterns</h2>
    <table>
        <tr>
            <th>Expected</th>
            <th>Predicted</th>
            <th>Count</th>
            <th>Example Queries</th>
        </tr>
"""
        
        for pattern in failures[:10]:
            examples = '<br>'.join(pattern['examples'][:3])
            html += f"""        <tr>
            <td>{pattern['expected']}</td>
            <td>{pattern['predicted']}</td>
            <td>{pattern['count']}</td>
            <td>{examples}</td>
        </tr>
"""
        
        html += """    </table>
</body>
</html>
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(html)
            logger.info(f"Report saved to {output_path}")
        
        return html


def run_quick_evaluation(
    max_conversations: int = 100,
    categories: List[str] = None,
) -> Dict[str, Any]:
    """
    Run a quick evaluation with sensible defaults.
    
    Args:
        max_conversations: Maximum conversations to evaluate
        categories: Optional list of categories to test
        
    Returns:
        Evaluation summary dict
    """
    runner = ExperimentRunner()
    
    # Ensure we have conversations
    runner.ensure_conversations(1000)
    
    # Create config
    config = ExperimentConfig(
        name=f"quick_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Quick evaluation run",
        max_conversations=max_conversations,
        conversation_categories=categories or [],
        use_llm_metrics=False,
    )
    
    # Run experiment
    exp_id = runner.run_experiment(config)
    
    # Get experiment results
    exp = runner.db.get_experiment(exp_id)
    
    return {
        'experiment_id': exp_id,
        'total': exp.total_conversations,
        'passed': exp.passed_conversations,
        'pass_rate': exp.passed_conversations / exp.total_conversations if exp.total_conversations else 0,
        'intent_accuracy': exp.overall_intent_accuracy,
        'entity_f1': exp.overall_entity_f1,
        'tool_accuracy': exp.overall_tool_accuracy,
        'latency_ms': exp.avg_latency_ms,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Bioinformatics Agent Evaluation System")
    print("=" * 60)
    
    # Run quick evaluation
    result = run_quick_evaluation(max_conversations=50)
    
    print(f"\nResults:")
    print(f"  Pass Rate: {result['pass_rate']*100:.1f}%")
    print(f"  Intent Accuracy: {result['intent_accuracy']*100:.1f}%")
    print(f"  Entity F1: {result['entity_f1']*100:.1f}%")
    print(f"  Tool Accuracy: {result['tool_accuracy']*100:.1f}%")
    print(f"  Avg Latency: {result['latency_ms']:.0f}ms")
    
    # Generate HTML report
    runner = ExperimentRunner()
    runner.generate_html_report(
        result['experiment_id'],
        Path(__file__).parent / "reports" / f"{result['experiment_id']}.html"
    )
