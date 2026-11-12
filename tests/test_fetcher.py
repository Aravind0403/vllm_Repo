import json
from datetime import datetime, timezone, timedelta

import pytest

from pr_radar.fetcher import GitHubFetcher, GRAPHQL_QUERY
from helpers import create_sample_pr

REPO = "test/repo"
NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)


def _write_legacy_cache(fetcher, prs):
    with open(fetcher.cache_path(REPO), "w") as f:
        json.dump([p.model_dump(mode="json") for p in prs], f)


def _node(number, updated_at, title="t"):
    return {
        "number": number,
        "title": title,
        "body": "",
        "author": {"login": "alice"},
        "url": f"https://github.com/{REPO}/pull/{number}",
        "isDraft": False,
        "baseRefName": "main",
        "headRefName": f"feat-{number}",
        "createdAt": (updated_at - timedelta(days=3)).isoformat().replace("+00:00", "Z"),
        "updatedAt": updated_at.isoformat().replace("+00:00", "Z"),
        "reviewDecision": None,
        "mergeable": "MERGEABLE",
        "commits": {"nodes": [{"commit": {"committedDate": updated_at.isoformat(), "statusCheckRollup": {"state": "SUCCESS"}}}]},
        "labels": {"nodes": []},
        "files": {"nodes": [{"path": "a.py", "additions": 1, "deletions": 1}]},
        "reviewRequests": {"nodes": [{"requestedReviewer": {"login": "bob"}}, {"requestedReviewer": {"name": "core-team"}}]},
        "reviews": {"nodes": [{"author": {"login": "carol"}, "state": "APPROVED", "submittedAt": updated_at.isoformat()}]},
        "comments": {"nodes": [{"author": {"login": "alice"}, "createdAt": updated_at.isoformat()}]},
    }


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_legacy_list_cache_loads_and_infers_sync_time(tmp_path):
    f = GitHubFetcher(cache_dir=str(tmp_path), offline=True)
    older = create_sample_pr(1, updated_at=NOW - timedelta(days=5))
    newer = create_sample_pr(2, updated_at=NOW - timedelta(days=1))
    _write_legacy_cache(f, [older, newer])

    prs, synced_at = f.load_cache(REPO)
    assert {p.number for p in prs} == {1, 2}
    assert synced_at == newer.updated_at


def test_fresh_cache_short_circuits_without_network(tmp_path, monkeypatch):
    f = GitHubFetcher(cache_dir=str(tmp_path), offline=False)
    f.save_cache(REPO, [create_sample_pr(1)], datetime.now(timezone.utc))

    def boom(*a, **k):
        raise AssertionError("network must not be called")
    monkeypatch.setattr(f, "_fetch_from_github", boom)

    prs = f.fetch_open_prs(REPO, max_age_minutes=30)
    assert [p.number for p in prs] == [1]
    assert f.last_sync_info["mode"] == "cache"


def test_stale_cache_triggers_incremental_merge_and_eviction(tmp_path, monkeypatch):
    f = GitHubFetcher(cache_dir=str(tmp_path), offline=False)
    synced = datetime.now(timezone.utc) - timedelta(hours=2)
    cached = [
        create_sample_pr(1, title="unchanged", updated_at=synced - timedelta(days=1)),
        create_sample_pr(2, title="old version", updated_at=synced - timedelta(days=1)),
        create_sample_pr(3, title="got merged", updated_at=synced - timedelta(days=1)),
    ]
    f.save_cache(REPO, cached, synced)

    seen = {}

    def fake_fetch(owner, repo, total_limit, stop_before=None):
        seen["stop_before"] = stop_before
        seen["limit"] = total_limit
        return [create_sample_pr(2, title="new version"), create_sample_pr(4, title="brand new")]

    def fake_resolve(repo_slug, numbers):
        seen["resolved"] = sorted(numbers)
        return {1: {"state": "OPEN"}, 3: {"state": "MERGED"}}

    monkeypatch.setattr(f, "_fetch_from_github", fake_fetch)
    monkeypatch.setattr(f, "resolve_target_prs", fake_resolve)

    prs = f.fetch_open_prs(REPO, limit=0, max_age_minutes=30)
    by_num = {p.number: p for p in prs}

    assert seen["stop_before"] == synced
    assert seen["limit"] == 0
    assert seen["resolved"] == [1, 3]
    assert set(by_num) == {1, 2, 4}
    assert by_num[2].title == "new version"
    info = f.last_sync_info
    assert info["mode"] == "incremental" and info["fetched"] == 2 and info["evicted"] == 1

    # Cache was rewritten in the new format with a fresh timestamp
    _, synced_after = f.load_cache(REPO)
    assert synced_after > synced


def test_failed_refresh_falls_back_to_cache(tmp_path, monkeypatch):
    f = GitHubFetcher(cache_dir=str(tmp_path), offline=False)
    f.save_cache(REPO, [create_sample_pr(7)], datetime.now(timezone.utc) - timedelta(days=1))

    def boom(*a, **k):
        raise RuntimeError("github down")
    monkeypatch.setattr(f, "_fetch_from_github", boom)

    prs = f.fetch_open_prs(REPO, max_age_minutes=30)
    assert [p.number for p in prs] == [7]
    assert f.last_sync_info["mode"] == "cache"
    assert "refresh failed" in f.last_sync_info["note"]


def test_offline_without_cache_returns_empty(tmp_path):
    f = GitHubFetcher(cache_dir=str(tmp_path), offline=True)
    assert f.fetch_open_prs(REPO) == []
    assert f.last_sync_info["mode"] == "cache"


def test_full_fetch_paginates_everything_and_parses_review_fields(tmp_path, monkeypatch):
    f = GitHubFetcher(cache_dir=str(tmp_path), offline=False)
    f.token = "x"
    pages = [
        {"data": {"repository": {"pullRequests": {
            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
            "nodes": [_node(10, NOW), _node(9, NOW - timedelta(hours=1))]}}}},
        {"data": {"repository": {"pullRequests": {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": [_node(8, NOW - timedelta(hours=2))]}}}},
    ]
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json["variables"])
        return FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr("pr_radar.fetcher.requests.post", fake_post)

    prs = f.fetch_open_prs(REPO, limit=0, use_cache=False)
    assert [p.number for p in prs] == [10, 9, 8]
    assert len(calls) == 2 and calls[1]["cursor"] == "c1"
    assert f.last_sync_info["mode"] == "full"

    pr = prs[0]
    assert pr.requested_reviewers == ["bob", "core-team"]
    assert pr.reviews[0].author == "carol" and pr.reviews[0].state == "APPROVED"
    assert pr.last_commit_at is not None
    assert pr.last_comment_author == "alice"

    # Cache is written in the new dict format
    raw = json.load(open(f.cache_path(REPO)))
    assert isinstance(raw, dict) and raw["repo"] == REPO and len(raw["prs"]) == 3


def test_incremental_fetch_stops_at_boundary(tmp_path, monkeypatch):
    f = GitHubFetcher(cache_dir=str(tmp_path), offline=False)
    f.token = "x"
    boundary = NOW - timedelta(hours=1)
    pages = [
        {"data": {"repository": {"pullRequests": {
            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
            "nodes": [_node(10, NOW), _node(9, boundary), _node(8, NOW - timedelta(days=1))]}}}},
    ]
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(1)
        return FakeResponse(pages[0])

    monkeypatch.setattr("pr_radar.fetcher.requests.post", fake_post)
    prs = f._fetch_from_github("test", "repo", 0, stop_before=boundary)
    assert [p.number for p in prs] == [10]
    assert len(calls) == 1


def test_heavy_page_halves_page_size_and_retries_same_cursor(tmp_path, monkeypatch):
    f = GitHubFetcher(cache_dir=str(tmp_path), offline=False)
    f.token = "x"
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json["variables"])
        if len(calls) == 1:
            return FakeResponse({}, status=502)
        return FakeResponse({"data": {"repository": {"pullRequests": {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": [_node(1, NOW)]}}}})

    monkeypatch.setattr("pr_radar.fetcher.requests.post", fake_post)
    prs = f._fetch_from_github("test", "repo", 0)
    assert [p.number for p in prs] == [1]
    assert calls[0]["limit"] == GitHubFetcher.PAGE_SIZE
    assert calls[1]["limit"] == GitHubFetcher.PAGE_SIZE // 2
    assert calls[0]["cursor"] == calls[1]["cursor"] is None
