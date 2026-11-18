from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Optional
from .models import PullRequest, ReviewEvent, ReviewFlow, ReviewerLoad


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _days_between(later: datetime, earlier: datetime) -> float:
    return max(0.0, (later - earlier).total_seconds() / 86400.0)


def _external_reviews(pr: PullRequest) -> List[ReviewEvent]:
    """Reviews by anyone other than the PR author, oldest first."""
    revs = [r for r in pr.reviews if r.author != pr.author and r.state != "PENDING"]
    revs.sort(key=lambda r: _utc(r.submitted_at) or datetime.min.replace(tzinfo=timezone.utc))
    return revs


def analyze_review_flow(
    pr: PullRequest,
    now: Optional[datetime] = None,
    blocked_by: Optional[List[int]] = None,
) -> ReviewFlow:
    """
    Decide whose move it is and how long the PR has been waiting for that move.
    Strictly reconciled with StatusTracker so triage and turn-tracking never disagree.

    waiting_on:
      AUTHOR    failing CI, merge conflict, changes requested, or reviewer commented after author's last push
      REVIEWER  non-draft, stack-unblocked, non-conflicting, CI not failing, and awaiting human review (incl. overdue/stale)
      MERGE     approved, non-draft, stack-unblocked, non-conflicting, and CI green/pending (exact match to READY_TO_MERGE)
      NONE      draft or blocked on an open upstream dependency PR
    """
    now = now or datetime.now(timezone.utc)
    reviews = _external_reviews(pr)
    requested = list(pr.requested_reviewers)
    requested_teams = list(pr.requested_teams)

    has_signals = bool(pr.reviews or requested or requested_teams or pr.last_commit_at or pr.last_comment_at)

    last_review = reviews[-1] if reviews else None
    last_review_at = _utc(last_review.submitted_at) if last_review else None

    author_events = [_utc(pr.created_at), _utc(pr.last_commit_at)]
    if pr.last_comment_author == pr.author:
        author_events.append(_utc(pr.last_comment_at))
    last_author_at = max((d for d in author_events if d is not None), default=_utc(pr.created_at))

    engaged = sorted({r.author for r in reviews})

    if pr.is_draft or blocked_by:
        waiting_on = "NONE"
        since = _utc(pr.updated_at)
    elif pr.mergeable == "CONFLICTING":
        waiting_on = "AUTHOR"
        since = _utc(pr.updated_at)
    elif pr.ci_status == "FAILURE":
        waiting_on = "AUTHOR"
        since = last_author_at or _utc(pr.updated_at)
    elif pr.review_decision == "CHANGES_REQUESTED":
        waiting_on = "AUTHOR"
        cr = [r for r in reviews if r.state == "CHANGES_REQUESTED"]
        since = _utc(cr[-1].submitted_at) if cr and cr[-1].submitted_at else (last_review_at or _utc(pr.updated_at))
    elif pr.review_decision == "APPROVED" and pr.ci_status in ("SUCCESS", "PENDING"):
        waiting_on = "MERGE"
        ap = [r for r in reviews if r.state == "APPROVED"]
        since = _utc(ap[-1].submitted_at) if ap and ap[-1].submitted_at else (last_review_at or _utc(pr.updated_at))
    elif last_review_at is not None and last_review_at > last_author_at and last_review.state in ("CHANGES_REQUESTED", "COMMENTED"):
        waiting_on = "AUTHOR"
        since = last_review_at
    else:
        waiting_on = "REVIEWER"
        candidates = [last_author_at, last_review_at]
        since = max(d for d in candidates if d is not None)

    if not has_signals:
        # Old cache without review data: fall back to updated_at as the only clock.
        since = _utc(pr.updated_at)

    return ReviewFlow(
        waiting_on=waiting_on,
        waiting_days=round(_days_between(now, since), 1),
        review_count=len(reviews),
        reviewers_engaged=engaged,
        requested_reviewers=requested,
        requested_teams=requested_teams,
        last_review_at=last_review_at,
        last_reviewer=last_review.author if last_review else None,
        last_review_state=last_review.state if last_review else None,
        last_author_activity_at=last_author_at,
        has_review_signals=has_signals,
    )


def is_pr_actionable_for_reviewer(
    pr: PullRequest,
    flow: Optional[ReviewFlow] = None,
    blocked_by: Optional[List[int]] = None,
) -> bool:
    """
    True iff a human reviewer can meaningfully review this PR right now:
    non-draft, stack-unblocked, non-conflicting, CI not failing, and waiting on REVIEWER.
    Stale/overdue PRs where the ball is still on the reviewer remain actionable.
    """
    if pr.is_draft or blocked_by:
        return False
    if pr.mergeable == "CONFLICTING" or pr.ci_status == "FAILURE":
        return False
    if flow is not None:
        return flow.waiting_on == "REVIEWER"
    return pr.review_decision not in ("CHANGES_REQUESTED", "APPROVED")


def compute_reviewer_load(
    prs: List[PullRequest],
    flows: Optional[Dict[int, ReviewFlow]] = None,
    analyses: Optional[Dict[int, object]] = None,
    now: Optional[datetime] = None,
) -> List[ReviewerLoad]:
    """
    Aggregate per-reviewer workload across open PRs, separating individual
    reviewers from @team/... requests and tracking Actionable Now vs Raw Pending.
    """
    now = now or datetime.now(timezone.utc)
    loads: Dict[str, ReviewerLoad] = {}

    def get(name: str, is_team: bool = False) -> ReviewerLoad:
        if name not in loads:
            loads[name] = ReviewerLoad(reviewer=name, is_team=is_team)
        return loads[name]

    for pr in prs:
        flow = flows.get(pr.number) if flows else None
        analysis = analyses.get(pr.number) if analyses else None
        blocked_by = getattr(analysis, "blocked_by", None) if analysis else None
        wait = flow.waiting_days if flow else 0.0
        idle_days = (
            getattr(analysis, "staleness_days", None)
            if analysis is not None
            else None
        )
        if idle_days is None:
            idle_days = _days_between(now, _utc(pr.updated_at) or now)

        actionable = is_pr_actionable_for_reviewer(pr, flow, blocked_by=blocked_by)
        is_stale_30d = idle_days >= 30.0 or wait >= 30.0
        is_actionable_stale_30d = actionable and is_stale_30d

        def _record_request(key: str, is_team: bool) -> None:
            rl = get(key, is_team=is_team)
            rl.pending_requests += 1
            rl.pending_pr_numbers.append(pr.number)
            rl.oldest_pending_days = max(rl.oldest_pending_days, wait)
            if is_stale_30d:
                rl.stale_pending_30d += 1
            if actionable:
                rl.actionable_pending += 1
                rl.actionable_pr_numbers.append(pr.number)
                rl.oldest_actionable_days = max(rl.oldest_actionable_days, wait)
                if is_actionable_stale_30d:
                    rl.actionable_stale_30d += 1

        for name in pr.requested_reviewers:
            _record_request(name, is_team=False)
        for team in pr.requested_teams:
            _record_request(f"@team/{team}", is_team=True)

        latest_by_reviewer: Dict[str, ReviewEvent] = {}
        for r in _external_reviews(pr):
            latest_by_reviewer[r.author] = r   # sorted oldest first, so last wins
        for name, r in latest_by_reviewer.items():
            rl = get(name, is_team=False)
            rl.reviews_submitted += 1
            if r.state == "APPROVED":
                rl.approvals += 1
            elif r.state == "CHANGES_REQUESTED":
                rl.changes_requested += 1

    result = list(loads.values())
    for rl in result:
        rl.pending_pr_numbers.sort()
        rl.actionable_pr_numbers.sort()
    result.sort(
        key=lambda r: (r.actionable_pending, r.pending_requests, r.reviews_submitted),
        reverse=True,
    )
    return result
