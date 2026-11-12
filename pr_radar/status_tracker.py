from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Optional, Set
from .models import PullRequest, TriageStatus, PRAnalysis
from .dag_engine import DAGEngine
from .collision_detector import CollisionDetector


class StatusTracker:
    """
    Evaluates PR status, CI health, review bottlenecks, staleness,
    and dependency blockers to give a unified triage diagnosis for every PR.
    """

    STALE_THRESHOLD_DAYS = 14.0

    @classmethod
    def analyze_pr(
        cls,
        pr: PullRequest,
        dag: DAGEngine,
        open_pr_numbers: Set[int],
        now: Optional[datetime] = None,
    ) -> PRAnalysis:
        if now is None:
            now = datetime.now(timezone.utc)

        # Ensure tz-aware comparison
        pr_updated = pr.updated_at
        if pr_updated.tzinfo is None:
            pr_updated = pr_updated.replace(tzinfo=timezone.utc)

        staleness_days = max(0.0, (now - pr_updated).total_seconds() / 86400.0)

        # Blockers and dependents from DAG
        direct_blockers = dag.get_direct_blockers(pr.number)
        direct_dependents = dag.get_direct_dependents(pr.number)

        # Filter blockers that are still OPEN
        active_blockers = [b for b in direct_blockers if b in open_pr_numbers]

        # Determine Triage Status
        status = TriageStatus.WAITING_FOR_REVIEW
        primary_reason = "Awaiting maintainer review"

        if pr.is_draft:
            status = TriageStatus.DRAFT
            primary_reason = "Marked as draft / work in progress"

        elif active_blockers:
            status = TriageStatus.BLOCKED_BY_DEPENDENCY
            blocker_list = ", ".join(f"#{b}" for b in active_blockers)
            primary_reason = f"Blocked on open dependency: {blocker_list}"

        elif pr.mergeable == "CONFLICTING":
            status = TriageStatus.BLOCKED_BY_CONFLICT
            primary_reason = "Merge conflict with base branch"

        elif pr.ci_status == "FAILURE":
            status = TriageStatus.BLOCKED_BY_CI
            primary_reason = "CI checks are failing"

        elif pr.review_decision == "CHANGES_REQUESTED":
            status = TriageStatus.WAITING_FOR_AUTHOR
            primary_reason = "Reviewer requested changes"

        elif pr.review_decision == "APPROVED":
            if pr.ci_status == "SUCCESS" and pr.mergeable == "MERGEABLE":
                status = TriageStatus.READY_TO_MERGE
                primary_reason = "Approved, CI green, mergeable"
            elif pr.ci_status == "PENDING":
                status = TriageStatus.READY_TO_MERGE
                primary_reason = "Approved, waiting for CI completion"
            else:
                status = TriageStatus.WAITING_FOR_REVIEW
                primary_reason = "Approved with pending requirements"

        elif staleness_days >= cls.STALE_THRESHOLD_DAYS:
            status = TriageStatus.STALE
            primary_reason = f"Inactive for {staleness_days:.1f} days"

        return PRAnalysis(
            pr=pr,
            triage_status=status,
            staleness_days=round(staleness_days, 1),
            blocked_by=active_blockers,
            blocks=direct_dependents,
            primary_blocker_reason=primary_reason,
        )

    @classmethod
    def analyze_all(
        cls,
        prs: List[PullRequest],
        dag: DAGEngine,
        now: Optional[datetime] = None,
    ) -> List[PRAnalysis]:
        open_pr_numbers = {pr.number for pr in prs if pr.state == "OPEN"}
        analyses = [
            cls.analyze_pr(pr, dag, open_pr_numbers, now)
            for pr in prs
        ]
        return analyses

