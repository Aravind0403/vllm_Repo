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


def analyze_review_flow(pr: PullRequest, now: Optional[datetime] = None) -> ReviewFlow:
    """
    Decide whose move it is and how long the PR has been waiting for that move.

    waiting_on:
      AUTHOR    reviewer asked for changes (or commented) after the author's last push/comment
      REVIEWER  author pushed/commented after the last review, or nobody has reviewed yet
      MERGE     approved and green; waiting on a maintainer to merge
      NONE      draft
    """
    now = now or datetime.now(timezone.utc)
    reviews = _external_reviews(pr)
    requested = list(pr.requested_reviewers)

    has_signals = bool(pr.reviews or requested or pr.last_commit_at or pr.last_comment_at)

    last_review = reviews[-1] if reviews else None
    last_review_at = _utc(last_review.submitted_at) if last_review else None

    author_events = [_utc(pr.created_at), _utc(pr.last_commit_at)]
    if pr.last_comment_author == pr.author:
        author_events.append(_utc(pr.last_comment_at))
    last_author_at = max((d for d in author_events if d is not None), default=_utc(pr.created_at))

    engaged = sorted({r.author for r in reviews})

    if pr.is_draft:
        waiting_on = "NONE"
        since = _utc(pr.updated_at)
    elif pr.review_decision == "CHANGES_REQUESTED":
        waiting_on = "AUTHOR"
        cr = [r for r in reviews if r.state == "CHANGES_REQUESTED"]
        since = _utc(cr[-1].submitted_at) if cr and cr[-1].submitted_at else (last_review_at or _utc(pr.updated_at))
    elif pr.review_decision == "APPROVED" and pr.ci_status != "FAILURE" and pr.mergeable != "CONFLICTING":
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
        last_review_at=last_review_at,
        last_reviewer=last_review.author if last_review else None,
        last_review_state=last_review.state if last_review else None,
        last_author_activity_at=last_author_at,
        has_review_signals=has_signals,
    )


def compute_reviewer_load(
    prs: List[PullRequest],
    flows: Optional[Dict[int, ReviewFlow]] = None,
) -> List[ReviewerLoad]:
    """
    Aggregate per-reviewer workload across open PRs.
    A pending request is a reviewer still listed in requested_reviewers (GitHub
    clears the request once they submit a review).
    """
    loads: Dict[str, ReviewerLoad] = {}

    def get(name: str) -> ReviewerLoad:
        if name not in loads:
            loads[name] = ReviewerLoad(reviewer=name)
        return loads[name]

    for pr in prs:
        flow = flows.get(pr.number) if flows else None
        wait = flow.waiting_days if flow else 0.0

        for name in pr.requested_reviewers:
            rl = get(name)
            rl.pending_requests += 1
            rl.pending_pr_numbers.append(pr.number)
            rl.oldest_pending_days = max(rl.oldest_pending_days, wait)

        latest_by_reviewer: Dict[str, ReviewEvent] = {}
        for r in _external_reviews(pr):
            latest_by_reviewer[r.author] = r   # sorted oldest first, so last wins
        for name, r in latest_by_reviewer.items():
            rl = get(name)
            rl.reviews_submitted += 1
            if r.state == "APPROVED":
                rl.approvals += 1
            elif r.state == "CHANGES_REQUESTED":
                rl.changes_requested += 1

    result = list(loads.values())
    for rl in result:
        rl.pending_pr_numbers.sort()
    result.sort(key=lambda r: (r.pending_requests, r.reviews_submitted), reverse=True)
    return result
