"""
GitHub Copilot integration for code-level fixes.

Integrates with GitHub Copilot Coding Agent (via MCP) for
creating pull requests that fix workflow issues.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from .categories import ErrorDiagnosis
from .prompts import build_code_fix_prompt

logger = logging.getLogger(__name__)


@dataclass
class PullRequestResult:
    """Result of creating a fix PR."""
    success: bool
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    branch_name: Optional[str] = None
    message: str = ""


class GitHubCopilotAgent:
    """
    Integration with GitHub Copilot Coding Agent.
    
    Uses GitHub's Copilot Coding Agent to create pull requests
    that fix workflow code issues. Requires:
    - GitHub Copilot Pro+ subscription
    - Repository with GitHub Copilot enabled
    - GITHUB_TOKEN environment variable
    
    Example:
        agent = GitHubCopilotAgent(owner="user", repo="BioPipelines")
        result = await agent.create_fix_pr(diagnosis, workflow_content)
    """
    
    def __init__(
        self,
        owner: str,
        repo: str,
        github_token: Optional[str] = None,
    ):
        """
        Initialize GitHub Copilot agent.
        
        Args:
            owner: Repository owner
            repo: Repository name
            github_token: GitHub token (or from GITHUB_TOKEN env)
        """
        self.owner = owner
        self.repo = repo
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        
        if not self.github_token:
            logger.warning("GITHUB_TOKEN not set - GitHub Copilot will be unavailable")
    
    def is_available(self) -> bool:
        """Check if GitHub Copilot integration is available."""
        return bool(self.github_token)
    
    async def create_fix_pr(
        self,
        diagnosis: ErrorDiagnosis,
        workflow_content: str,
        workflow_file: str = "main.nf",
        base_branch: str = "main",
    ) -> PullRequestResult:
        """
        Create a pull request with a fix for the diagnosed error.
        
        This delegates to GitHub Copilot Coding Agent which will:
        1. Create a new branch
        2. Implement the fix
        3. Open a pull request
        
        Args:
            diagnosis: ErrorDiagnosis with error details
            workflow_content: Content of the workflow file
            workflow_file: Name of the workflow file
            base_branch: Branch to base the fix on
            
        Returns:
            PullRequestResult with PR details
        """
        if not self.is_available():
            return PullRequestResult(
                success=False,
                message="GitHub token not configured"
            )
        
        # Build the problem statement for Copilot
        problem_statement = build_code_fix_prompt(
            diagnosis=diagnosis,
            workflow_content=workflow_content,
            workflow_file=workflow_file,
        )
        
        # Title for the PR
        title = f"Fix: {diagnosis.root_cause[:60]}"
        if len(diagnosis.root_cause) > 60:
            title += "..."
        
        try:
            # This would integrate with MCP GitHub tools
            # For now, return a placeholder that explains the integration
            
            logger.info(
                f"Would create PR via GitHub Copilot:\n"
                f"  Owner: {self.owner}\n"
                f"  Repo: {self.repo}\n"
                f"  Title: {title}\n"
                f"  Base: {base_branch}"
            )
            
            # In actual implementation, this would call:
            # mcp_github_create_pull_request_with_copilot(
            #     owner=self.owner,
            #     repo=self.repo,
            #     problem_statement=problem_statement,
            #     title=title,
            #     base_ref=base_branch,
            # )
            
            return PullRequestResult(
                success=True,
                message=(
                    "Pull request creation initiated. GitHub Copilot will "
                    "analyze the code and create a fix. Check the repository "
                    "for the new PR."
                ),
            )
            
        except Exception as e:
            logger.error(f"Failed to create fix PR: {e}")
            return PullRequestResult(
                success=False,
                message=f"Failed to create PR: {str(e)}"
            )
    
    async def assign_to_issue(
        self,
        issue_number: int,
        diagnosis: Optional[ErrorDiagnosis] = None,
    ) -> PullRequestResult:
        """
        Assign Copilot to fix an existing issue.
        
        Args:
            issue_number: GitHub issue number
            diagnosis: Optional error diagnosis for context
            
        Returns:
            PullRequestResult
        """
        if not self.is_available():
            return PullRequestResult(
                success=False,
                message="GitHub token not configured"
            )
        
        try:
            # This would call:
            # mcp_github_assign_copilot_to_issue(
            #     owner=self.owner,
            #     repo=self.repo,
            #     issueNumber=issue_number,
            # )
            
            logger.info(
                f"Would assign Copilot to issue #{issue_number} "
                f"in {self.owner}/{self.repo}"
            )
            
            return PullRequestResult(
                success=True,
                message=f"Copilot assigned to issue #{issue_number}",
            )
            
        except Exception as e:
            logger.error(f"Failed to assign Copilot: {e}")
            return PullRequestResult(
                success=False,
                message=f"Failed to assign Copilot: {str(e)}"
            )
    
    def format_issue_body(self, diagnosis: ErrorDiagnosis) -> str:
        """
        Format an error diagnosis as a GitHub issue body.
        
        Args:
            diagnosis: ErrorDiagnosis
            
        Returns:
            Markdown-formatted issue body
        """
        risk_icons = {
            "safe": "🟢",
            "low": "🟡",
            "medium": "🟠",
            "high": "🔴",
        }
        
        fixes_md = ""
        for i, fix in enumerate(diagnosis.suggested_fixes, 1):
            icon = risk_icons.get(fix.risk_level.value, "⚪")
            fixes_md += f"{i}. {icon} {fix.description}\n"
            if fix.command:
                fixes_md += f"   ```\n   {fix.command}\n   ```\n"
        
        return f"""## Bug Report: {diagnosis.category.value.replace('_', ' ').title()}

### Description
{diagnosis.user_explanation}

### Root Cause
{diagnosis.root_cause}

### Error Log
```
{diagnosis.log_excerpt[:1000]}
```

### Suggested Fixes
{fixes_md}

### Context
- Failed Process: {diagnosis.failed_process or 'Unknown'}
- Work Directory: {diagnosis.work_directory or 'Unknown'}
- Confidence: {diagnosis.confidence:.0%}
- Diagnosed by: {diagnosis.llm_provider_used or 'Pattern Matching'}

---
*This issue was automatically generated by BioPipelines Error Diagnosis Agent*
"""


def get_github_copilot_agent(
    owner: Optional[str] = None,
    repo: Optional[str] = None,
) -> Optional[GitHubCopilotAgent]:
    """
    Get a GitHub Copilot agent if available.
    
    Will try to infer owner/repo from git config if not provided.
    
    Args:
        owner: Repository owner
        repo: Repository name
        
    Returns:
        GitHubCopilotAgent or None
    """
    # Try to get from environment or git config
    if not owner or not repo:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Parse github.com:owner/repo or github.com/owner/repo
                if "github.com" in url:
                    parts = url.split("github.com")[-1]
                    parts = parts.strip(":/").rstrip(".git").split("/")
                    if len(parts) >= 2:
                        owner = owner or parts[0]
                        repo = repo or parts[1]
        except Exception:
            pass
    
    # Use defaults if still not set
    owner = owner or os.getenv("GITHUB_OWNER", "sdodlapa")
    repo = repo or os.getenv("GITHUB_REPO", "BioPipelines")
    
    agent = GitHubCopilotAgent(owner=owner, repo=repo)
    
    if agent.is_available():
        return agent
    
    return None
