#!/usr/bin/env python3
"""
Historical Tracking Database for BioPipelines Evaluation

Provides SQLite-based storage for:
- Evaluation run history with trends
- Per-category metric tracking over time
- Regression detection with historical context
- Baseline version management
- Smart test selection based on change impact

Usage:
    from tests.evaluation.historical_tracker import HistoricalTracker
    
    tracker = HistoricalTracker()
    tracker.save_run(evaluation_report)
    trends = tracker.get_trends(days=30)
    regressions = tracker.detect_historical_regressions()
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import hashlib
import os

# Database location
DB_PATH = Path(__file__).parent.parent.parent / "reports" / "evaluation" / "history.db"


@dataclass
class EvaluationRun:
    """Single evaluation run record."""
    run_id: str
    timestamp: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    total_tests: int
    intent_accuracy: float
    entity_f1: float
    tool_accuracy: float
    avg_latency_ms: float
    llm_usage_rate: float
    category_metrics: dict
    regressions: list
    is_baseline: bool = False
    notes: Optional[str] = None


@dataclass
class CategoryTrend:
    """Trend data for a category."""
    category: str
    current_accuracy: float
    previous_accuracy: float
    trend_7d: float  # % change over 7 days
    trend_30d: float  # % change over 30 days
    best_accuracy: float
    worst_accuracy: float
    avg_accuracy: float
    samples: int


@dataclass  
class RegressionEvent:
    """Historical regression event."""
    run_id: str
    timestamp: str
    metric: str
    baseline_value: float
    regressed_value: float
    severity: str
    recovered: bool
    recovery_run_id: Optional[str] = None


class HistoricalTracker:
    """SQLite-based historical tracking for evaluation metrics."""
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            self.db_path = DB_PATH
        elif isinstance(db_path, str):
            self.db_path = Path(db_path)
        else:
            self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Main evaluation runs table
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    git_commit TEXT,
                    git_branch TEXT,
                    total_tests INTEGER NOT NULL,
                    intent_accuracy REAL NOT NULL,
                    entity_f1 REAL NOT NULL,
                    tool_accuracy REAL NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    llm_usage_rate REAL NOT NULL,
                    category_metrics TEXT NOT NULL,  -- JSON
                    regressions TEXT NOT NULL,  -- JSON
                    is_baseline INTEGER DEFAULT 0,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Category-level metrics for efficient trend queries
                CREATE TABLE IF NOT EXISTS category_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    intent_accuracy REAL NOT NULL,
                    entity_f1 REAL NOT NULL,
                    tool_accuracy REAL NOT NULL,
                    total_tests INTEGER NOT NULL,
                    error_count INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id)
                );
                
                -- Index for efficient trend queries
                CREATE INDEX IF NOT EXISTS idx_category_metrics_category 
                ON category_metrics(category, timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_category_metrics_timestamp
                ON category_metrics(timestamp);
                
                -- Regression events for tracking
                CREATE TABLE IF NOT EXISTS regression_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    baseline_value REAL NOT NULL,
                    regressed_value REAL NOT NULL,
                    severity TEXT NOT NULL,
                    recovered INTEGER DEFAULT 0,
                    recovery_run_id TEXT,
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id)
                );
                
                -- Test case results for smart selection
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    test_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    query TEXT NOT NULL,
                    expected_intent TEXT NOT NULL,
                    actual_intent TEXT NOT NULL,
                    intent_correct INTEGER NOT NULL,
                    entity_f1 REAL NOT NULL,
                    parse_time_ms REAL NOT NULL,
                    is_flaky INTEGER DEFAULT 0,  -- Inconsistent results
                    consecutive_failures INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_test_results_test_id
                ON test_results(test_id);
                
                -- File change impact tracking
                CREATE TABLE IF NOT EXISTS change_impact (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    category TEXT NOT NULL,
                    impact_score REAL NOT NULL,  -- 0-1, how much this file affects this category
                    last_updated TEXT NOT NULL
                );
                
                CREATE UNIQUE INDEX IF NOT EXISTS idx_change_impact_file_category
                ON change_impact(file_path, category);
                
                -- Baselines table
                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reason TEXT,
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id)
                );
            """)
    
    def _get_git_info(self) -> tuple[Optional[str], Optional[str]]:
        """Get current git commit and branch."""
        try:
            import subprocess
            
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.db_path.parent.parent.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()[:8]
            
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.db_path.parent.parent.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            return commit, branch
        except Exception:
            return None, None
    
    def save_run(self, report: dict, notes: Optional[str] = None) -> str:
        """
        Save an evaluation run to the database.
        
        Args:
            report: EvaluationReport dict from unified_evaluation_runner
            notes: Optional notes about this run
            
        Returns:
            run_id: Unique ID for this run
        """
        timestamp = report.get("timestamp", datetime.now().isoformat())
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"
        
        git_commit, git_branch = self._get_git_info()
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert main run record
            conn.execute("""
                INSERT INTO evaluation_runs (
                    run_id, timestamp, git_commit, git_branch,
                    total_tests, intent_accuracy, entity_f1, tool_accuracy,
                    avg_latency_ms, llm_usage_rate, category_metrics, regressions,
                    is_baseline, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                timestamp,
                git_commit,
                git_branch,
                report.get("total_tests", 0),
                report.get("overall_intent_accuracy", 0.0),
                report.get("overall_entity_f1", 0.0),
                report.get("overall_tool_accuracy", 0.0),
                report.get("overall_avg_latency_ms", 0.0),
                report.get("overall_llm_usage_rate", 0.0),
                json.dumps(report.get("category_metrics", {})),
                json.dumps(report.get("regressions", [])),
                0,
                notes
            ))
            
            # Insert category metrics
            for category, metrics in report.get("category_metrics", {}).items():
                conn.execute("""
                    INSERT INTO category_metrics (
                        run_id, category, intent_accuracy, entity_f1,
                        tool_accuracy, total_tests, error_count, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    category,
                    metrics.get("intent_accuracy", 0.0),
                    metrics.get("entity_f1_avg", 0.0),
                    metrics.get("tool_accuracy", 0.0),
                    metrics.get("total_tests", 0),
                    metrics.get("error_count", 0),
                    timestamp
                ))
            
            # Insert test results
            for result in report.get("test_results", []):
                conn.execute("""
                    INSERT INTO test_results (
                        run_id, test_id, category, query, expected_intent,
                        actual_intent, intent_correct, entity_f1, parse_time_ms, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    result.get("test_id", ""),
                    result.get("category", ""),
                    result.get("query", "")[:500],  # Truncate long queries
                    result.get("expected_intent", ""),
                    result.get("actual_intent", ""),
                    1 if result.get("intent_correct", False) else 0,
                    result.get("entity_f1", 0.0),
                    result.get("parse_time_ms", 0.0),
                    timestamp
                ))
            
            # Track regressions
            for regression in report.get("regressions", []):
                conn.execute("""
                    INSERT INTO regression_events (
                        run_id, timestamp, metric, baseline_value,
                        regressed_value, severity
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    timestamp,
                    regression.get("metric", ""),
                    regression.get("baseline", 0.0),
                    regression.get("current", 0.0),
                    regression.get("severity", "medium")
                ))
            
            conn.commit()
        
        return run_id
    
    def get_recent_runs(self, limit: int = 20) -> list[EvaluationRun]:
        """Get recent evaluation runs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM evaluation_runs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [
                EvaluationRun(
                    run_id=row["run_id"],
                    timestamp=row["timestamp"],
                    git_commit=row["git_commit"],
                    git_branch=row["git_branch"],
                    total_tests=row["total_tests"],
                    intent_accuracy=row["intent_accuracy"],
                    entity_f1=row["entity_f1"],
                    tool_accuracy=row["tool_accuracy"],
                    avg_latency_ms=row["avg_latency_ms"],
                    llm_usage_rate=row["llm_usage_rate"],
                    category_metrics=json.loads(row["category_metrics"]),
                    regressions=json.loads(row["regressions"]),
                    is_baseline=bool(row["is_baseline"]),
                    notes=row["notes"]
                )
                for row in rows
            ]
    
    def get_category_trends(self, category: str, days: int = 30) -> CategoryTrend:
        """Get trend data for a specific category."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get all metrics in range
            rows = conn.execute("""
                SELECT intent_accuracy, timestamp
                FROM category_metrics
                WHERE category = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (category, cutoff_date)).fetchall()
            
            if not rows:
                return CategoryTrend(
                    category=category,
                    current_accuracy=0.0,
                    previous_accuracy=0.0,
                    trend_7d=0.0,
                    trend_30d=0.0,
                    best_accuracy=0.0,
                    worst_accuracy=0.0,
                    avg_accuracy=0.0,
                    samples=0
                )
            
            accuracies = [row["intent_accuracy"] for row in rows]
            current = accuracies[0]
            previous = accuracies[1] if len(accuracies) > 1 else current
            
            # Calculate 7-day trend
            cutoff_7d = (datetime.now() - timedelta(days=7)).isoformat()
            recent = [row["intent_accuracy"] for row in rows if row["timestamp"] > cutoff_7d]
            older = [row["intent_accuracy"] for row in rows if row["timestamp"] <= cutoff_7d]
            
            avg_recent = sum(recent) / len(recent) if recent else current
            avg_older = sum(older) / len(older) if older else previous
            
            trend_7d = (avg_recent - avg_older) / avg_older * 100 if avg_older > 0 else 0
            
            # 30-day trend is current vs oldest in range
            oldest = accuracies[-1] if accuracies else current
            trend_30d = (current - oldest) / oldest * 100 if oldest > 0 else 0
            
            return CategoryTrend(
                category=category,
                current_accuracy=current,
                previous_accuracy=previous,
                trend_7d=trend_7d,
                trend_30d=trend_30d,
                best_accuracy=max(accuracies),
                worst_accuracy=min(accuracies),
                avg_accuracy=sum(accuracies) / len(accuracies),
                samples=len(rows)
            )
    
    def get_all_trends(self, days: int = 30) -> dict[str, CategoryTrend]:
        """Get trends for all categories."""
        with sqlite3.connect(self.db_path) as conn:
            categories = conn.execute("""
                SELECT DISTINCT category FROM category_metrics
            """).fetchall()
            
            return {
                cat[0]: self.get_category_trends(cat[0], days)
                for cat in categories
            }
    
    def detect_historical_regressions(self, lookback_runs: int = 5) -> list[dict]:
        """
        Detect regressions by comparing recent runs to historical patterns.
        
        More sophisticated than simple baseline comparison:
        - Considers variance in historical data
        - Detects gradual degradation
        - Identifies flaky tests
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get recent runs
            recent = conn.execute("""
                SELECT * FROM evaluation_runs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (lookback_runs + 1,)).fetchall()
            
            if len(recent) < 2:
                return []
            
            regressions = []
            current = recent[0]
            historical = recent[1:]
            
            # Calculate historical stats
            hist_intent = [r["intent_accuracy"] for r in historical]
            hist_entity = [r["entity_f1"] for r in historical]
            
            import statistics
            
            if len(hist_intent) >= 2:
                mean_intent = statistics.mean(hist_intent)
                std_intent = statistics.stdev(hist_intent)
                
                # Regression if current is more than 2 std below mean
                if current["intent_accuracy"] < mean_intent - 2 * std_intent:
                    regressions.append({
                        "type": "statistical",
                        "metric": "intent_accuracy",
                        "current": current["intent_accuracy"],
                        "historical_mean": mean_intent,
                        "historical_std": std_intent,
                        "z_score": (current["intent_accuracy"] - mean_intent) / std_intent if std_intent > 0 else 0,
                        "severity": "high" if current["intent_accuracy"] < mean_intent - 3 * std_intent else "medium"
                    })
            
            # Detect gradual degradation (consistent decrease over runs)
            if len(hist_intent) >= 3:
                consecutive_decreases = 0
                for i in range(len(historical) - 1):
                    if historical[i]["intent_accuracy"] < historical[i+1]["intent_accuracy"]:
                        consecutive_decreases += 1
                    else:
                        break
                
                if consecutive_decreases >= 3:
                    regressions.append({
                        "type": "gradual_degradation",
                        "metric": "intent_accuracy",
                        "consecutive_decreases": consecutive_decreases,
                        "trend": hist_intent[0] - hist_intent[-1],
                        "severity": "medium"
                    })
            
            return regressions
    
    def get_flaky_tests(self, min_runs: int = 5, flaky_threshold: float = 0.3) -> list[dict]:
        """
        Identify tests with inconsistent results.
        
        A test is considered flaky if it has high variance in pass/fail.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get tests with enough history
            results = conn.execute("""
                SELECT test_id, category, query,
                       COUNT(*) as run_count,
                       SUM(intent_correct) as pass_count,
                       AVG(intent_correct) as pass_rate
                FROM test_results
                GROUP BY test_id
                HAVING run_count >= ?
            """, (min_runs,)).fetchall()
            
            flaky = []
            for row in results:
                pass_rate = row["pass_rate"]
                # Flaky if pass rate is between 30% and 70%
                if flaky_threshold < pass_rate < (1 - flaky_threshold):
                    flaky.append({
                        "test_id": row["test_id"],
                        "category": row["category"],
                        "query": row["query"][:100],
                        "run_count": row["run_count"],
                        "pass_rate": pass_rate,
                        "flakiness": 1 - abs(pass_rate - 0.5) * 2  # 0-1, higher = more flaky
                    })
            
            return sorted(flaky, key=lambda x: x["flakiness"], reverse=True)
    
    def get_smart_test_selection(
        self,
        changed_files: list[str],
        max_tests: int = 50
    ) -> list[str]:
        """
        Select tests to run based on changed files.
        
        Uses change_impact table to prioritize tests most likely affected.
        """
        if not changed_files:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get impacted categories
            placeholders = ",".join("?" * len(changed_files))
            impacts = conn.execute(f"""
                SELECT category, SUM(impact_score) as total_impact
                FROM change_impact
                WHERE file_path IN ({placeholders})
                GROUP BY category
                ORDER BY total_impact DESC
            """, changed_files).fetchall()
            
            if not impacts:
                # No impact data, fall back to recently failed tests
                return self._get_recently_failed_tests(max_tests)
            
            # Get tests from impacted categories
            categories = [row["category"] for row in impacts]
            category_placeholders = ",".join("?" * len(categories))
            
            tests = conn.execute(f"""
                SELECT DISTINCT test_id
                FROM test_results
                WHERE category IN ({category_placeholders})
                ORDER BY RANDOM()
                LIMIT ?
            """, categories + [max_tests]).fetchall()
            
            return [row["test_id"] for row in tests]
    
    def _get_recently_failed_tests(self, limit: int) -> list[str]:
        """Get tests that failed in recent runs."""
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT DISTINCT test_id
                FROM test_results
                WHERE intent_correct = 0
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [row[0] for row in results]
    
    def update_change_impact(self, file_path: str, category: str, impact_score: float):
        """Update the impact score for a file-category pair."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO change_impact (file_path, category, impact_score, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_path, category) DO UPDATE SET
                    impact_score = ?,
                    last_updated = ?
            """, (
                file_path, category, impact_score, datetime.now().isoformat(),
                impact_score, datetime.now().isoformat()
            ))
            conn.commit()
    
    def mark_as_baseline(self, run_id: str, reason: Optional[str] = None):
        """Mark a run as a baseline."""
        with sqlite3.connect(self.db_path) as conn:
            # Unmark previous baselines
            conn.execute("UPDATE evaluation_runs SET is_baseline = 0")
            
            # Mark this run
            conn.execute("""
                UPDATE evaluation_runs SET is_baseline = 1 WHERE run_id = ?
            """, (run_id,))
            
            # Record in baselines table
            conn.execute("""
                INSERT INTO baselines (run_id, created_at, reason)
                VALUES (?, ?, ?)
            """, (run_id, datetime.now().isoformat(), reason))
            
            conn.commit()
    
    def get_baseline(self) -> Optional[EvaluationRun]:
        """Get the current baseline run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM evaluation_runs
                WHERE is_baseline = 1
                LIMIT 1
            """).fetchone()
            
            if not row:
                return None
            
            return EvaluationRun(
                run_id=row["run_id"],
                timestamp=row["timestamp"],
                git_commit=row["git_commit"],
                git_branch=row["git_branch"],
                total_tests=row["total_tests"],
                intent_accuracy=row["intent_accuracy"],
                entity_f1=row["entity_f1"],
                tool_accuracy=row["tool_accuracy"],
                avg_latency_ms=row["avg_latency_ms"],
                llm_usage_rate=row["llm_usage_rate"],
                category_metrics=json.loads(row["category_metrics"]),
                regressions=json.loads(row["regressions"]),
                is_baseline=True,
                notes=row["notes"]
            )
    
    def generate_trend_report(self, days: int = 30) -> dict:
        """Generate a comprehensive trend report."""
        trends = self.get_all_trends(days)
        recent_runs = self.get_recent_runs(limit=10)
        flaky_tests = self.get_flaky_tests()
        historical_regressions = self.detect_historical_regressions()
        
        # Calculate overall trends
        if recent_runs:
            intent_values = [r.intent_accuracy for r in recent_runs]
            entity_values = [r.entity_f1 for r in recent_runs]
            
            overall_intent_trend = (intent_values[0] - intent_values[-1]) / intent_values[-1] * 100 if intent_values[-1] > 0 else 0
            overall_entity_trend = (entity_values[0] - entity_values[-1]) / entity_values[-1] * 100 if entity_values[-1] > 0 else 0
        else:
            overall_intent_trend = 0
            overall_entity_trend = 0
        
        return {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "overall_trends": {
                "intent_accuracy_trend": overall_intent_trend,
                "entity_f1_trend": overall_entity_trend,
                "direction": "improving" if overall_intent_trend > 0 else "declining" if overall_intent_trend < 0 else "stable"
            },
            "category_trends": {
                cat: {
                    "current": t.current_accuracy,
                    "trend_7d": t.trend_7d,
                    "trend_30d": t.trend_30d,
                    "best": t.best_accuracy,
                    "samples": t.samples
                }
                for cat, t in trends.items()
            },
            "recent_runs": [
                {
                    "run_id": r.run_id,
                    "timestamp": r.timestamp,
                    "intent_accuracy": r.intent_accuracy,
                    "entity_f1": r.entity_f1,
                    "git_commit": r.git_commit
                }
                for r in recent_runs[:5]
            ],
            "flaky_tests_count": len(flaky_tests),
            "flaky_tests": flaky_tests[:10],
            "regressions": historical_regressions,
            "health_score": self._calculate_health_score(trends, flaky_tests, historical_regressions)
        }
    
    def _calculate_health_score(
        self,
        trends: dict[str, CategoryTrend],
        flaky_tests: list[dict],
        regressions: list[dict]
    ) -> dict:
        """Calculate overall health score (0-100)."""
        score = 100
        issues = []
        
        # Penalize for regressions
        for reg in regressions:
            if reg.get("severity") == "high":
                score -= 20
                issues.append(f"High severity regression in {reg.get('metric')}")
            else:
                score -= 10
                issues.append(f"Medium severity regression in {reg.get('metric')}")
        
        # Penalize for declining trends
        for cat, trend in trends.items():
            if trend.trend_7d < -5:  # More than 5% decline in 7 days
                score -= 5
                issues.append(f"{cat} declining by {abs(trend.trend_7d):.1f}% over 7 days")
        
        # Penalize for flaky tests
        if len(flaky_tests) > 10:
            score -= 10
            issues.append(f"{len(flaky_tests)} flaky tests detected")
        elif len(flaky_tests) > 5:
            score -= 5
            issues.append(f"{len(flaky_tests)} flaky tests detected")
        
        score = max(0, min(100, score))
        
        if score >= 90:
            status = "excellent"
        elif score >= 70:
            status = "good"
        elif score >= 50:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "score": score,
            "status": status,
            "issues": issues
        }


# Convenience function
def get_tracker() -> HistoricalTracker:
    """Get the default historical tracker instance."""
    return HistoricalTracker()


if __name__ == "__main__":
    # Test the tracker
    tracker = HistoricalTracker()
    
    # Generate a sample report
    sample_report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": 100,
        "overall_intent_accuracy": 0.85,
        "overall_entity_f1": 0.72,
        "overall_tool_accuracy": 0.90,
        "overall_avg_latency_ms": 450.0,
        "overall_llm_usage_rate": 0.15,
        "category_metrics": {
            "data_discovery": {
                "intent_accuracy": 0.90,
                "entity_f1_avg": 0.75,
                "tool_accuracy": 0.92,
                "total_tests": 15,
                "error_count": 1
            }
        },
        "regressions": [],
        "test_results": []
    }
    
    run_id = tracker.save_run(sample_report, notes="Test run")
    print(f"Saved run: {run_id}")
    
    # Get trend report
    report = tracker.generate_trend_report()
    print(f"Health score: {report['health_score']}")
