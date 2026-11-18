from datetime import datetime, timezone, timedelta

from pr_radar.models import ReviewEvent
from pr_radar.review_flow import analyze_review_flow, compute_reviewer_load
from helpers import create_sample_pr

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)


def _rev(author, state, days_ago):
    return ReviewEvent(author=author, state=state, submitted_at=NOW - timedelta(days=days_ago))


def test_no_review_signals_falls_back_to_updated_at():
    pr = create_sample_pr(1, updated_at=NOW - timedelta(days=6))
    flow = analyze_review_flow(pr, NOW)
    assert flow.has_review_signals is False
    assert flow.waiting_on == "REVIEWER"
    assert flow.waiting_days == 6.0


def test_changes_requested_waits_on_author_since_that_review():
    pr = create_sample_pr(
        2, review_decision="CHANGES_REQUESTED", updated_at=NOW,
        reviews=[_rev("bob", "COMMENTED", 10), _rev("bob", "CHANGES_REQUESTED", 4)],
        last_commit_at=NOW - timedelta(days=8),
    )
    flow = analyze_review_flow(pr, NOW)
    assert flow.waiting_on == "AUTHOR"
    assert flow.waiting_days == 4.0
    assert flow.last_reviewer == "bob" and flow.last_review_state == "CHANGES_REQUESTED"


def test_author_push_after_review_hands_ball_back_to_reviewer():
    pr = create_sample_pr(
        3, review_decision="REVIEW_REQUIRED", updated_at=NOW,
        reviews=[_rev("bob", "COMMENTED", 5)],
        last_commit_at=NOW - timedelta(days=2),
        requested_reviewers=["bob"],
    )
    flow = analyze_review_flow(pr, NOW)
    assert flow.waiting_on == "REVIEWER"
    assert flow.waiting_days == 2.0
    assert flow.requested_reviewers == ["bob"]


def test_reviewer_comment_after_push_waits_on_author():
    pr = create_sample_pr(
        4, review_decision=None, updated_at=NOW,
        reviews=[_rev("bob", "COMMENTED", 1)],
        last_commit_at=NOW - timedelta(days=3),
    )
    flow = analyze_review_flow(pr, NOW)
    assert flow.waiting_on == "AUTHOR"
    assert flow.waiting_days == 1.0


def test_approved_and_green_waits_on_merge():
    pr = create_sample_pr(5, review_decision="APPROVED", ci_status="SUCCESS",
                          reviews=[_rev("carol", "APPROVED", 3)], updated_at=NOW)
    flow = analyze_review_flow(pr, NOW)
    assert flow.waiting_on == "MERGE" and flow.waiting_days == 3.0


def test_draft_waits_on_nobody_and_author_self_reviews_ignored():
    pr = create_sample_pr(6, is_draft=True, author="alice", reviews=[_rev("alice", "COMMENTED", 1)], updated_at=NOW)
    flow = analyze_review_flow(pr, NOW)
    assert flow.waiting_on == "NONE"
    assert flow.review_count == 0 and flow.reviewers_engaged == []


def test_reviewer_load_aggregation():
    prs = [
        create_sample_pr(1, requested_reviewers=["bob", "carol"], updated_at=NOW - timedelta(days=7)),
        create_sample_pr(2, requested_reviewers=["bob"], reviews=[_rev("dave", "APPROVED", 1)], updated_at=NOW - timedelta(days=2)),
        create_sample_pr(3, reviews=[_rev("bob", "COMMENTED", 5), _rev("bob", "CHANGES_REQUESTED", 2)], updated_at=NOW),
    ]
    flows = {p.number: analyze_review_flow(p, NOW) for p in prs}
    loads = {rl.reviewer: rl for rl in compute_reviewer_load(prs, flows)}

    assert loads["bob"].pending_requests == 2
    assert loads["bob"].pending_pr_numbers == [1, 2]
    # PR 1 has no commits or reviews, so it has waited since creation (updated - 2d = 9d)
    assert loads["bob"].oldest_pending_days == 9.0
    assert loads["bob"].reviews_submitted == 1 and loads["bob"].changes_requested == 1
    assert loads["carol"].pending_requests == 1
    assert loads["dave"].approvals == 1 and loads["dave"].pending_requests == 0
    # Sorted by actionable_pending, then pending first
    assert compute_reviewer_load(prs, flows)[0].reviewer == "bob"


def test_ci_failure_and_conflict_wait_on_author_and_unknown_mergeable_still_ready():
    pr_ci = create_sample_pr(10, ci_status="FAILURE", updated_at=NOW)
    assert analyze_review_flow(pr_ci, NOW).waiting_on == "AUTHOR"

    pr_conf = create_sample_pr(11, mergeable="CONFLICTING", updated_at=NOW)
    assert analyze_review_flow(pr_conf, NOW).waiting_on == "AUTHOR"

    pr_blocked = create_sample_pr(12, updated_at=NOW)
    assert analyze_review_flow(pr_blocked, NOW, blocked_by=[99]).waiting_on == "NONE"

    pr_unknown_mergeable = create_sample_pr(
        13, review_decision="APPROVED", ci_status="SUCCESS", mergeable="UNKNOWN",
        reviews=[_rev("carol", "APPROVED", 1)], updated_at=NOW,
    )
    assert analyze_review_flow(pr_unknown_mergeable, NOW).waiting_on == "MERGE"


def test_reviewer_load_separates_actionable_stale_and_teams():
    pr_ok = create_sample_pr(20, requested_reviewers=["alice_rev"], updated_at=NOW - timedelta(days=2))
    pr_overdue = create_sample_pr(21, requested_reviewers=["alice_rev"], updated_at=NOW - timedelta(days=45))
    pr_ci_fail = create_sample_pr(22, requested_reviewers=["alice_rev"], ci_status="FAILURE", updated_at=NOW - timedelta(days=40))
    pr_draft = create_sample_pr(23, requested_reviewers=["alice_rev"], is_draft=True, updated_at=NOW - timedelta(days=5))
    pr_team = create_sample_pr(24, updated_at=NOW - timedelta(days=1))
    pr_team.requested_teams = ["vllm-core"]

    prs = [pr_ok, pr_overdue, pr_ci_fail, pr_draft, pr_team]
    flows = {p.number: analyze_review_flow(p, NOW) for p in prs}
    loads = {rl.reviewer: rl for rl in compute_reviewer_load(prs, flows, now=NOW)}

    alice = loads["alice_rev"]
    assert alice.is_team is False
    assert alice.pending_requests == 4
    assert alice.actionable_pending == 2          # pr_ok + pr_overdue (overdue stays actionable!)
    assert alice.actionable_stale_30d == 1        # pr_overdue badged as >30d overdue
    assert alice.stale_pending_30d == 2           # pr_overdue + pr_ci_fail

    team = loads["@team/vllm-core"]
    assert team.is_team is True
    assert team.actionable_pending == 1
