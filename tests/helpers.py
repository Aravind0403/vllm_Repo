from datetime import datetime, timezone, timedelta
from typing import List, Optional

from pr_radar.models import PullRequest, ChangedFile, ReviewEvent


def create_sample_pr(
    number: int,
    title: str = "Sample PR",
    body: str = "",
    files: Optional[List[str]] = None,
    base_ref: str = "main",
    head_ref: str = "",
    review_decision: Optional[str] = None,
    mergeable: str = "MERGEABLE",
    ci_status: str = "SUCCESS",
    is_draft: bool = False,
    updated_at: Optional[datetime] = None,
    repo: str = "test/repo",
    author: str = "alice",
    additions: int = 10,
    deletions: int = 5,
    requested_reviewers: Optional[List[str]] = None,
    reviews: Optional[List[ReviewEvent]] = None,
    last_commit_at: Optional[datetime] = None,
    last_comment_at: Optional[datetime] = None,
    last_comment_author: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> PullRequest:
    if updated_at is None:
        updated_at = datetime.now(timezone.utc)
    file_objs = [ChangedFile(path=f, additions=additions, deletions=deletions) for f in (files or [])]
    return PullRequest(
        repo=repo,
        number=number,
        title=title,
        body=body,
        author=author,
        url=f"https://github.com/{repo}/pull/{number}",
        is_draft=is_draft,
        base_ref=base_ref,
        head_ref=head_ref,
        created_at=created_at or (updated_at - timedelta(days=2)),
        updated_at=updated_at,
        review_decision=review_decision,
        mergeable=mergeable,
        ci_status=ci_status,
        files=file_objs,
        requested_reviewers=requested_reviewers or [],
        reviews=reviews or [],
        last_commit_at=last_commit_at,
        last_comment_at=last_comment_at,
        last_comment_author=last_comment_author,
    )
