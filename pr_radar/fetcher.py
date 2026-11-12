from __future__ import annotations

import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests

from .models import PullRequest, ChangedFile, ReviewEvent


GRAPHQL_QUERY = """
query($owner: String!, $repo: String!, $limit: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(first: $limit, after: $cursor, states: OPEN, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        body
        author {
          login
        }
        url
        isDraft
        baseRefName
        headRefName
        createdAt
        updatedAt
        reviewDecision
        mergeable
        commits(last: 1) {
          nodes {
            commit {
              committedDate
              statusCheckRollup {
                state
              }
            }
          }
        }
        labels(first: 10) {
          nodes {
            name
          }
        }
        files(first: 100) {
          nodes {
            path
            additions
            deletions
          }
        }
        reviewRequests(first: 10) {
          nodes {
            requestedReviewer {
              ... on User { login }
              ... on Team { name }
            }
          }
        }
        reviews(last: 30) {
          nodes {
            author { login }
            state
            submittedAt
          }
        }
        comments(last: 1) {
          nodes {
            author { login }
            createdAt
          }
        }
      }
    }
  }
}
"""

OFFLINE_ENV = "PR_RADAR_OFFLINE"


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class GitHubFetcher:
    """
    Fetches open PRs from GitHub via GraphQL with a local JSON cache.

    Cache file layout (per repo): {"synced_at": iso-utc, "repo": slug, "prs": [...]}.
    Older caches that are a bare list are still readable; their sync time is taken
    as the newest `updated_at` they contain.

    Refresh modes:
      - "cache":       cache present and younger than max_age -> no network
      - "incremental": cache present but stale -> fetch only PRs updated since the
                       last sync, merge them in, and evict PRs that are no longer open
      - "full":        no cache (or use_cache=False) -> paginate the whole open set
    Set PR_RADAR_OFFLINE=1 to forbid network calls entirely (cache or empty).
    """

    PAGE_SIZE = 25        # GitHub 502s on heavier pages (files + reviews per PR)
    MIN_PAGE_SIZE = 5
    RETRYABLE_STATUS = (502, 503, 504)

    def __init__(self, cache_dir: str = "data", offline: Optional[bool] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = (
            offline if offline is not None
            else os.environ.get(OFFLINE_ENV, "").strip().lower() in ("1", "true", "yes")
        )
        self.token = None if self.offline else self._resolve_token()
        self.last_synced_at: Optional[datetime] = None
        self.last_sync_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------ auth
    def _resolve_token(self) -> Optional[str]:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            return token
        try:
            res = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return None

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # ----------------------------------------------------------------- cache
    def cache_path(self, repo_slug: str) -> Path:
        owner, repo = repo_slug.split("/", 1)
        return self.cache_dir / f"{owner}_{repo}_prs.json"

    def load_cache(self, repo_slug: str) -> Tuple[List[PullRequest], Optional[datetime]]:
        """Returns (prs, synced_at). synced_at is None when there is no cache."""
        path = self.cache_path(repo_slug)
        if not path.exists():
            return [], None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return [], None

        if isinstance(raw, dict):
            items = raw.get("prs", [])
            synced_at = _parse_ts(raw.get("synced_at"))
        else:
            items = raw
            synced_at = None

        prs: List[PullRequest] = []
        for item in items:
            try:
                prs.append(PullRequest(**item))
            except Exception:
                continue

        if synced_at is None:
            # Legacy list cache: the newest updated_at is a lower bound on sync time.
            synced_at = max((_utc(p.updated_at) for p in prs), default=None)
        return prs, synced_at

    def cache_synced_at(self, repo_slug: str) -> Optional[datetime]:
        _, synced_at = self.load_cache(repo_slug)
        return synced_at

    def save_cache(self, repo_slug: str, prs: List[PullRequest], synced_at: datetime) -> None:
        path = self.cache_path(repo_slug)
        payload = {
            "synced_at": synced_at.isoformat(),
            "repo": repo_slug,
            "prs": [pr.model_dump(mode="json") for pr in prs],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    # ----------------------------------------------------------------- fetch
    def fetch_open_prs(
        self,
        repo_slug: str,
        limit: int = 500,
        use_cache: bool = True,
        max_age_minutes: Optional[float] = 30.0,
    ) -> List[PullRequest]:
        """
        Fetch open PRs for 'owner/name'.

        limit: cap on PRs returned (0 = no cap). On a full fetch it also caps how
               many PRs are pulled from GitHub.
        use_cache: False forces a full re-fetch.
        max_age_minutes: cache older than this triggers an incremental refresh;
                         None means a present cache is always accepted as-is.
        """
        now = datetime.now(timezone.utc)
        cached, synced_at = self.load_cache(repo_slug) if use_cache else ([], None)
        has_cache = synced_at is not None

        if has_cache:
            age_min = (now - synced_at).total_seconds() / 60.0
            fresh = max_age_minutes is None or age_min <= max_age_minutes
            if fresh or self.offline:
                self._record("cache", synced_at, len(cached), 0, 0,
                             note="offline" if (self.offline and not fresh) else "")
                return self._apply_limit(cached, limit)

            try:
                merged, fetched, evicted = self._fetch_incremental(repo_slug, cached, synced_at, now)
            except Exception as e:
                print(f"Warning: incremental refresh failed, serving cache: {e}")
                self._record("cache", synced_at, len(cached), 0, 0, note=f"refresh failed: {e}")
                return self._apply_limit(cached, limit)

            self.save_cache(repo_slug, merged, now)
            self._record("incremental", now, len(merged), fetched, evicted)
            return self._apply_limit(merged, limit)

        if self.offline:
            self._record("cache", None, 0, 0, 0, note="offline, no cache")
            return []

        owner, repo = repo_slug.split("/", 1)
        prs = self._fetch_from_github(owner, repo, limit)
        self.save_cache(repo_slug, prs, now)
        self._record("full", now, len(prs), len(prs), 0)
        return prs

    def _record(self, mode: str, synced_at: Optional[datetime], total: int,
                fetched: int, evicted: int, note: str = "") -> None:
        self.last_synced_at = synced_at
        self.last_sync_info = {
            "mode": mode,
            "synced_at": synced_at.isoformat() if synced_at else None,
            "total": total,
            "fetched": fetched,
            "evicted": evicted,
            "note": note,
        }

    @staticmethod
    def _apply_limit(prs: List[PullRequest], limit: int) -> List[PullRequest]:
        prs = sorted(prs, key=lambda p: _utc(p.updated_at), reverse=True)
        return prs[:limit] if limit and limit > 0 else prs

    def _fetch_incremental(
        self,
        repo_slug: str,
        cached: List[PullRequest],
        synced_at: datetime,
        now: datetime,
    ) -> Tuple[List[PullRequest], int, int]:
        """Returns (merged_prs, fetched_count, evicted_count)."""
        owner, repo = repo_slug.split("/", 1)
        updated = self._fetch_from_github(owner, repo, 0, stop_before=synced_at)

        merged: Dict[int, PullRequest] = {pr.number: pr for pr in cached}
        for pr in updated:
            merged[pr.number] = pr

        # Anything we did not see may have been merged/closed since the last sync.
        unseen = [n for n in merged if n not in {p.number for p in updated}]
        evicted = 0
        if unseen:
            states = self.resolve_target_prs(repo_slug, unseen)
            for num in unseen:
                info = states.get(num)
                if info is not None and info.get("state") != "OPEN":
                    merged.pop(num, None)
                    evicted += 1

        return list(merged.values()), len(updated), evicted

    def _fetch_from_github(
        self,
        owner: str,
        repo: str,
        total_limit: int,
        stop_before: Optional[datetime] = None,
    ) -> List[PullRequest]:
        """
        Paginate open PRs ordered by updatedAt desc.
        total_limit 0 = unlimited. stop_before stops paging once a node is not
        newer than that timestamp (incremental mode).
        """
        if self.offline:
            raise RuntimeError("network disabled (PR_RADAR_OFFLINE)")

        api_url = "https://api.github.com/graphql"
        prs: List[PullRequest] = []
        cursor: Optional[str] = None
        unlimited = not total_limit or total_limit <= 0
        remaining = total_limit
        page_size = self.PAGE_SIZE

        while unlimited or remaining > 0:
            batch_size = page_size if unlimited else min(remaining, page_size)
            variables = {"owner": owner, "repo": repo, "limit": batch_size, "cursor": cursor}

            resp = requests.post(
                api_url,
                json={"query": GRAPHQL_QUERY, "variables": variables},
                headers=self._headers(),
                timeout=60,
            )
            if resp.status_code in self.RETRYABLE_STATUS and page_size > self.MIN_PAGE_SIZE:
                # Page too heavy for GitHub to compute in time: shrink and retry the same cursor.
                page_size = max(self.MIN_PAGE_SIZE, page_size // 2)
                continue
            if resp.status_code != 200:
                raise RuntimeError(
                    f"GitHub GraphQL query failed ({resp.status_code}): {resp.text[:300]}"
                )

            try:
                data = resp.json()
            except ValueError:
                # Truncated / HTML body on a 200: treat like a heavy page and retry smaller.
                if page_size > self.MIN_PAGE_SIZE:
                    page_size = max(self.MIN_PAGE_SIZE, page_size // 2)
                    continue
                raise RuntimeError("GitHub returned a non-JSON response")
            if "errors" in data and not data.get("data"):
                raise RuntimeError(f"GraphQL error: {data['errors']}")

            repo_data = (data.get("data") or {}).get("repository")
            if not repo_data:
                break

            pr_data = repo_data.get("pullRequests", {})
            nodes = pr_data.get("nodes", [])

            reached_boundary = False
            for node in nodes:
                if stop_before is not None:
                    node_updated = _parse_ts(node.get("updatedAt"))
                    if node_updated is not None and node_updated <= stop_before:
                        reached_boundary = True
                        break
                prs.append(self._convert_node_to_pr(f"{owner}/{repo}", node))

            if reached_boundary:
                break

            page_info = pr_data.get("pageInfo", {})
            cursor = page_info.get("endCursor")
            if not unlimited:
                remaining -= len(nodes)
            if not page_info.get("hasNextPage", False) or not nodes:
                break

        return prs

    def resolve_target_prs(
        self,
        repo_slug: str,
        pr_numbers: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Batch query state/title of specific PR numbers.
        Returns: {pr_number: {'state': 'OPEN'|'MERGED'|'CLOSED', 'title': ..., 'isDraft': bool}}
        Numbers that are not PRs (e.g. issues) are absent from the result.
        """
        if not pr_numbers or self.offline:
            return {}

        owner, repo = repo_slug.split("/", 1)
        results: Dict[int, Dict[str, Any]] = {}
        for i in range(0, len(pr_numbers), 50):
            chunk = pr_numbers[i : i + 50]
            field_queries = [
                f"pr_{num}: pullRequest(number: {num}) {{ state isDraft title url }}"
                for num in chunk
            ]
            query_str = f"""
            query {{
              repository(owner: "{owner}", name: "{repo}") {{
                {' '.join(field_queries)}
              }}
            }}
            """
            try:
                resp = requests.post(
                    "https://api.github.com/graphql",
                    json={"query": query_str},
                    headers=self._headers(),
                    timeout=20,
                )
                if resp.status_code == 200:
                    data = (resp.json().get("data") or {}).get("repository") or {}
                    for key, val in data.items():
                        if key.startswith("pr_") and val:
                            results[int(key.split("_")[1])] = val
            except Exception as e:
                print(f"Warning: Failed to resolve target PRs: {e}")

        return results

    # --------------------------------------------------------------- convert
    def _convert_node_to_pr(self, repo_slug: str, node: Dict[str, Any]) -> PullRequest:
        commits = (node.get("commits") or {}).get("nodes") or []
        ci_status = "UNKNOWN"
        last_commit_at: Optional[datetime] = None
        if commits:
            commit = (commits[0] or {}).get("commit") or {}
            rollup = commit.get("statusCheckRollup")
            if rollup:
                ci_status = rollup.get("state", "UNKNOWN")
            last_commit_at = _parse_ts(commit.get("committedDate"))

        file_nodes = (node.get("files") or {}).get("nodes") or []
        files = [
            ChangedFile(
                path=fn.get("path", ""),
                additions=fn.get("additions", 0),
                deletions=fn.get("deletions", 0),
            )
            for fn in file_nodes
            if fn.get("path")
        ]

        label_nodes = (node.get("labels") or {}).get("nodes") or []
        labels = [ln.get("name", "") for ln in label_nodes if ln.get("name")]

        author = node.get("author")
        author_login = author.get("login") if author else "ghost"

        requested: List[str] = []
        for rr in (node.get("reviewRequests") or {}).get("nodes") or []:
            who = (rr or {}).get("requestedReviewer") or {}
            name = who.get("login") or who.get("name")
            if name:
                requested.append(name)

        reviews: List[ReviewEvent] = []
        for rv in (node.get("reviews") or {}).get("nodes") or []:
            if not rv:
                continue
            rv_author = (rv.get("author") or {}).get("login") or "ghost"
            reviews.append(
                ReviewEvent(
                    author=rv_author,
                    state=rv.get("state", "COMMENTED"),
                    submitted_at=_parse_ts(rv.get("submittedAt")),
                )
            )

        last_comment_at: Optional[datetime] = None
        last_comment_author: Optional[str] = None
        comment_nodes = (node.get("comments") or {}).get("nodes") or []
        if comment_nodes and comment_nodes[0]:
            last_comment_at = _parse_ts(comment_nodes[0].get("createdAt"))
            last_comment_author = (comment_nodes[0].get("author") or {}).get("login")

        return PullRequest(
            repo=repo_slug,
            number=node["number"],
            title=node["title"],
            body=node.get("body") or "",
            author=author_login,
            url=node["url"],
            is_draft=node.get("isDraft", False),
            base_ref=node.get("baseRefName", "main"),
            head_ref=node.get("headRefName", ""),
            created_at=_parse_ts(node["createdAt"]),
            updated_at=_parse_ts(node["updatedAt"]),
            state="OPEN",
            review_decision=node.get("reviewDecision"),
            mergeable=node.get("mergeable", "UNKNOWN"),
            ci_status=ci_status,
            files=files,
            labels=labels,
            requested_reviewers=requested,
            reviews=reviews,
            last_commit_at=last_commit_at,
            last_comment_at=last_comment_at,
            last_comment_author=last_comment_author,
        )
