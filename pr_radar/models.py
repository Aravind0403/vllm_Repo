from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DependencyKind(str, Enum):
    STACKED_ON = "stacked_on"
    DEPENDS_ON = "depends_on"
    BLOCKED_BY = "blocked_by"
    REQUIRES = "requires"
    BRANCH_BASE = "branch_base"
    RELATED = "related"


class TriageStatus(str, Enum):
    READY_TO_MERGE = "READY_TO_MERGE"          # Approved, CI pass, mergeable, no blockers
    BLOCKED_BY_DEPENDENCY = "BLOCKED_BY_DEP"   # Waiting on another PR
    BLOCKED_BY_CONFLICT = "CONFLICTING"        # Merge conflict with main
    BLOCKED_BY_CI = "CI_FAILURE"               # CI failed
    WAITING_FOR_REVIEW = "WAITING_REVIEW"      # Needs reviewer attention
    WAITING_FOR_AUTHOR = "CHANGES_REQUESTED"   # Reviewer requested changes
    DRAFT = "DRAFT"                            # Author marked as draft
    STALE = "STALE"                            # Inactive for > 14 days


class ChangedFile(BaseModel):
    path: str
    additions: int = 0
    deletions: int = 0

    @property
    def changes(self) -> int:
        return self.additions + self.deletions


class ReviewEvent(BaseModel):
    """A single submitted review on a PR."""
    author: str
    state: str                     # APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED, PENDING
    submitted_at: Optional[datetime] = None


class DependencyRelation(BaseModel):
    source_pr: int
    target_pr: int
    kind: DependencyKind
    context_snippet: str = ""
    confidence: float = 1.0


class PullRequest(BaseModel):
    repo: str
    number: int
    title: str
    body: Optional[str] = ""
    author: str
    url: str
    is_draft: bool = False
    base_ref: str = "main"
    head_ref: str = ""
    created_at: datetime
    updated_at: datetime
    state: str = "OPEN"
    review_decision: Optional[str] = None      # APPROVED, CHANGES_REQUESTED, REVIEW_REQUIRED, None
    mergeable: str = "UNKNOWN"                 # MERGEABLE, CONFLICTING, UNKNOWN
    ci_status: str = "UNKNOWN"                 # SUCCESS, FAILURE, PENDING, UNKNOWN
    files: List[ChangedFile] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    assignees: List[str] = Field(default_factory=list)

    # Review-flow signals (empty/None when loaded from an older cache)
    requested_reviewers: List[str] = Field(default_factory=list)
    reviews: List[ReviewEvent] = Field(default_factory=list)
    last_commit_at: Optional[datetime] = None
    last_comment_at: Optional[datetime] = None
    last_comment_author: Optional[str] = None

    @property
    def file_paths(self) -> set[str]:
        return {f.path for f in self.files}

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.files)

    @property
    def total_changes(self) -> int:
        return self.total_additions + self.total_deletions


class FileCollision(BaseModel):
    pr_a: int
    pr_b: int
    overlapping_files: List[str]
    collision_score: float  # Normalized 0.0 - 1.0
    risk_level: str         # CRITICAL, HIGH, MEDIUM, LOW
    shared_churn: int = 0   # Total additions + deletions on overlapping files
    pr_a_title: str = ""
    pr_a_author: str = ""
    pr_a_idle_days: float = 0.0
    pr_b_title: str = ""
    pr_b_author: str = ""
    pr_b_idle_days: float = 0.0
    is_live_race: bool = False  # Both PRs updated recently (high collision risk)


class HotspotFile(BaseModel):
    path: str
    open_pr_count: int
    pr_numbers: List[int]
    total_churn: int
    min_idle_days: float = 0.0
    active_prs_count: int = 0  # PRs updated in last 5 days


class RiskScore(BaseModel):
    """Explainable change-risk assessment for a single PR."""
    score: int = 0                 # 0 - 100
    level: str = "LOW"             # CRITICAL, HIGH, MEDIUM, LOW
    reasons: List[str] = Field(default_factory=list)
    sensitive_areas: List[str] = Field(default_factory=list)
    sensitivity: float = 0.0       # component 0 - 1
    churn: float = 0.0             # component 0 - 1
    collision: float = 0.0         # component 0 - 1


class ReviewFlow(BaseModel):
    """Where the PR sits in the review loop and for how long."""
    waiting_on: str = "REVIEWER"           # REVIEWER, AUTHOR, NONE
    waiting_days: float = 0.0              # days since the ball last changed hands
    review_count: int = 0
    reviewers_engaged: List[str] = Field(default_factory=list)
    requested_reviewers: List[str] = Field(default_factory=list)
    last_review_at: Optional[datetime] = None
    last_reviewer: Optional[str] = None
    last_review_state: Optional[str] = None
    last_author_activity_at: Optional[datetime] = None
    has_review_signals: bool = False       # False when the PR came from an old cache


class ReviewerLoad(BaseModel):
    """Per-reviewer workload across all open PRs."""
    reviewer: str
    pending_requests: int = 0      # requested, not yet reviewed
    reviews_submitted: int = 0     # distinct PRs this reviewer has reviewed
    approvals: int = 0
    changes_requested: int = 0
    pending_pr_numbers: List[int] = Field(default_factory=list)
    oldest_pending_days: float = 0.0


class PRAnalysis(BaseModel):
    pr: PullRequest
    triage_status: TriageStatus
    staleness_days: float
    blocked_by: List[int] = Field(default_factory=list)
    blocks: List[int] = Field(default_factory=list)
    collision_prs: List[int] = Field(default_factory=list)
    primary_blocker_reason: Optional[str] = None
    risk: Optional[RiskScore] = None
    review_flow: Optional[ReviewFlow] = None
