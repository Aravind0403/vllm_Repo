from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .engine import PRRadarEngine
from .models import TriageStatus

DEFAULT_REPO = os.environ.get("PR_RADAR_REPO", "vllm-project/vllm")
SYNC_TTL_MINUTES = float(os.environ.get("PR_RADAR_SYNC_TTL_MINUTES", "30"))
SYNC_LIMIT = int(os.environ.get("PR_RADAR_SYNC_LIMIT", "0"))   # 0 = whole open set

_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_engines: Dict[str, PRRadarEngine] = {}
_refresh_attempts: Dict[str, datetime] = {}
_lock = threading.Lock()

app = FastAPI(
    title="PR Radar API",
    description="Live PR Dependency DAG, Collision Radar, Risk Scoring, and Review-Flow Engine",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_repo(repo: Optional[str]) -> str:
    slug = (repo or DEFAULT_REPO).strip()
    if not _REPO_RE.match(slug) or ".." in slug:
        raise HTTPException(status_code=400, detail=f"Invalid repo slug: {slug!r}")
    return slug


_refreshing: set = set()


def _background_refresh(slug: str) -> None:
    """Incremental GitHub refresh on a worker thread; swaps the engine in when done."""
    try:
        eng = PRRadarEngine(repo_slug=slug)
        eng.sync(limit=SYNC_LIMIT, use_cache=True, max_age_minutes=SYNC_TTL_MINUTES)
        with _lock:
            _engines[slug] = eng
    except Exception as e:  # keep serving the old engine
        print(f"Warning: background refresh for {slug} failed: {e}")
    finally:
        with _lock:
            _refreshing.discard(slug)


def get_engine(repo: Optional[str] = None) -> PRRadarEngine:
    """
    One engine per repo. First use loads the cache immediately (a full fetch
    only when there is no cache at all). Whenever the cache is older than
    SYNC_TTL_MINUTES an incremental refresh runs in the background, at most
    once per TTL window, and replaces the engine when it finishes.
    """
    slug = _resolve_repo(repo)
    now = datetime.now(timezone.utc)
    ttl = timedelta(minutes=SYNC_TTL_MINUTES)

    with _lock:
        eng = _engines.get(slug)
        if eng is None:
            eng = PRRadarEngine(repo_slug=slug)
            eng.sync(limit=SYNC_LIMIT, use_cache=True, max_age_minutes=None)
            _engines[slug] = eng

        stale = eng.last_synced_at is None or (now - eng.last_synced_at) > ttl
        last_try = _refresh_attempts.get(slug, datetime.min.replace(tzinfo=timezone.utc))
        if stale and (now - last_try) >= ttl and not eng.fetcher.offline and slug not in _refreshing:
            _refresh_attempts[slug] = now
            _refreshing.add(slug)
            threading.Thread(target=_background_refresh, args=(slug,), daemon=True).start()
        return eng


def is_refreshing(slug: str) -> bool:
    with _lock:
        return slug in _refreshing


def _regex(subsystem: Optional[str]) -> Optional[re.Pattern]:
    if not subsystem or subsystem == "ALL":
        return None
    try:
        return re.compile(subsystem, re.IGNORECASE)
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"Invalid subsystem regex: {e}")


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.get("/api/v1/health")
def get_health(repo: Optional[str] = Query(None)) -> Dict[str, Any]:
    eng = get_engine(repo)
    return {
        "status": "healthy",
        "repo": eng.repo_slug,
        "prs_indexed": len(eng.prs),
        "last_synced_at": eng.last_synced_at.isoformat() if eng.last_synced_at else None,
        "sync_mode": eng.sync_info.get("mode"),
        "sync_note": eng.sync_info.get("note", ""),
        "offline": eng.fetcher.offline,
        "refreshing": is_refreshing(eng.repo_slug),
        "sync_ttl_minutes": SYNC_TTL_MINUTES,
        "version": "1.1.0",
    }


@app.get("/api/v1/overview")
def get_overview(repo: Optional[str] = Query(None)) -> Dict[str, Any]:
    eng = get_engine(repo)
    stats = eng.get_summary_stats()
    return {
        "repo": eng.repo_slug,
        "total_open_prs": stats["total_open_prs_analyzed"],
        "last_synced_at": stats["last_synced_at"],
        "sync_mode": stats["sync_mode"],
        "status_breakdown": stats["status_breakdown"],
        "foundational_unblockers": stats["foundational_unblockers"],
        "dependency_stacks_count": stats["dependency_stacks_count"],
        "critical_file_collisions": stats["critical_file_collisions"],
        "top_hotspot_files": stats["top_hotspot_files"],
        "risk_breakdown": stats["risk_breakdown"],
        "top_risk_prs": stats["top_risk_prs"],
        "waiting_on_breakdown": stats["waiting_on_breakdown"],
        "median_reviewer_wait_days": stats["median_reviewer_wait_days"],
        "top_reviewer_load": stats["top_reviewer_load"],
    }


@app.get("/api/v1/subsystems")
def get_subsystems(repo: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
    """Lists standard vLLM subsystems with count of matching active PRs."""
    eng = get_engine(repo)
    subsystem_definitions = [
        {"id": "V1", "name": "V1 Engine", "pattern": r"(?i)v1"},
        {"id": "ATTN", "name": "Attention & Kernels", "pattern": r"(?i)attention|kernel|flash|gdn|triton"},
        {"id": "ROCM", "name": "ROCm / AMD", "pattern": r"(?i)rocm|amd"},
        {"id": "HARDWARE", "name": "TPU & Hardware", "pattern": r"(?i)tpu|neuron|xpu|cuda"},
        {"id": "QUANT", "name": "Quantization", "pattern": r"(?i)quant|fp8|mxfp4|awq|gptq|quark"},
        {"id": "SPEC", "name": "Speculative Decoding", "pattern": r"(?i)spec|eagle|mtp|draft"},
        {"id": "DISTRIBUTED", "name": "Distributed / MoE", "pattern": r"(?i)moe|parallel|distributed"},
        {"id": "API", "name": "API & Serving", "pattern": r"(?i)serve|api|entrypoint|openai"},
    ]
    results = []
    for s in subsystem_definitions:
        regex = re.compile(s["pattern"])
        count = sum(1 for pr in eng.prs if regex.search(pr.title))
        results.append({**{k: s[k] for k in ("id", "name", "pattern")}, "matching_prs_count": count})
    return results


def _serialize_dag_node(eng: PRRadarEngine, num: int, is_base: bool = False) -> Dict[str, Any]:
    pr = eng.pr_map.get(num)
    if pr is not None:
        analysis = eng.analyses.get(num)
        return {
            "id": pr.number,
            "title": pr.title,
            "author": pr.author,
            "url": pr.url,
            "status": analysis.triage_status.value if analysis else "WAITING_REVIEW",
            "reason": analysis.primary_blocker_reason if analysis else "",
            "state": "OPEN",
            "ci_status": pr.ci_status,
            "review_decision": pr.review_decision or "NONE",
            "is_draft": pr.is_draft,
            "staleness_days": analysis.staleness_days if analysis else 0.0,
            "risk_level": analysis.risk.level if analysis and analysis.risk else "LOW",
            "risk_score": analysis.risk.score if analysis and analysis.risk else 0,
            "blocked_by": analysis.blocked_by if analysis else [],
            "blocks": analysis.blocks if analysis else [],
            "is_base": is_base,
            "is_external": False,
        }
    node_data = eng.dag.graph.nodes.get(num, {}) if eng.dag else {}
    state = node_data.get("state", "UNKNOWN")
    return {
        "id": num,
        "title": node_data.get("title", f"{'Upstream Base' if is_base else 'Stacked'} PR #{num}"),
        "author": node_data.get("author") or "external",
        "url": f"https://github.com/{eng.repo_slug}/pull/{num}",
        "status": state if is_base else "BLOCKED_BY_DEP",
        "reason": {
            "MERGED": "Already merged: dependents can rebase onto main",
            "CLOSED": "Closed without merging: dependents are orphaned",
            "OPEN": "Open but outside the synced window",
        }.get(state, "Not resolved (may be an issue number, not a PR)"),
        "state": state,
        "ci_status": "UNKNOWN",
        "review_decision": "NONE",
        "is_draft": node_data.get("is_draft", False) if is_base else False,
        "staleness_days": 0.0,
        "risk_level": "LOW",
        "risk_score": 0,
        "blocked_by": [],
        "blocks": [],
        "is_base": is_base,
        "is_external": True,
    }


@app.get("/api/v1/dag")
def get_dag(
    subsystem: Optional[str] = Query(None, description="Regex filter for subsystem"),
    scope: str = Query("stacks", description="'stacks' (only chained PRs) or 'all'"),
    repo: Optional[str] = Query(None),
) -> Dict[str, Any]:
    eng = get_engine(repo)
    chains = eng.dag.get_dependency_chains() if eng.dag else []
    cycles = eng.dag.get_cycles() if eng.dag else []

    chained_pr_nums = {num for chain in chains for num in chain}
    subsystem_regex = _regex(subsystem)

    nodes = []
    included_ids = set()

    for pr in eng.prs:
        if scope == "stacks" and pr.number not in chained_pr_nums:
            continue
        if subsystem_regex and not subsystem_regex.search(pr.title):
            continue
        nodes.append(_serialize_dag_node(eng, pr.number))
        included_ids.add(pr.number)

    matched_ids = set(included_ids)
    filtered_chains = [
        c for c in chains if not subsystem_regex or any(num in matched_ids for num in c)
    ]
    filtered_cycles = [
        c for c in cycles if not subsystem_regex or any(num in matched_ids for num in c)
    ]

    edges = []
    seen_edges = set()
    for chain in filtered_chains:
        for i in range(len(chain) - 1):
            base_num, waiting_num = chain[i], chain[i + 1]
            for num, is_base in ((base_num, True), (waiting_num, False)):
                if num in included_ids:
                    continue
                nodes.append(_serialize_dag_node(eng, num, is_base=is_base))
                included_ids.add(num)
            edge_key = (waiting_num, base_num)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({"from": waiting_num, "to": base_num, "kind": "stacked_on"})

    return {
        "repo": eng.repo_slug,
        "scope": scope,
        "nodes": nodes,
        "edges": edges,
        "chains": filtered_chains,
        "cycles": filtered_cycles,
    }


@app.get("/api/v1/collisions")
def get_collisions(
    subsystem: Optional[str] = Query(None),
    min_risk: Optional[str] = Query(None, description="CRITICAL, HIGH, MEDIUM, LOW"),
    live_only: bool = Query(False, description="Only collisions where both PRs are active"),
    file: Optional[str] = Query(None, description="Exact overlapping file path to filter by"),
    search: Optional[str] = Query(None, description="Search PR #, title, author, or file"),
    limit: int = Query(200),
    repo: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    eng = get_engine(repo)
    subsystem_regex = _regex(subsystem)
    search_lower = search.lower() if search else None
    results = []
    ordered = sorted(eng.collisions, key=lambda c: (not c.is_live_race, c.risk_level != "CRITICAL", -c.collision_score))
    for c in ordered:
        if min_risk and c.risk_level != min_risk:
            continue
        if live_only and not c.is_live_race:
            continue
        if file and file not in c.overlapping_files:
            continue
        if subsystem_regex:
            if not (subsystem_regex.search(c.pr_a_title) or subsystem_regex.search(c.pr_b_title)
                    or any(subsystem_regex.search(f) for f in c.overlapping_files)):
                continue
        if search_lower and not (
            search_lower in str(c.pr_a) or search_lower in str(c.pr_b)
            or search_lower in c.pr_a_title.lower() or search_lower in c.pr_b_title.lower()
            or search_lower in c.pr_a_author.lower() or search_lower in c.pr_b_author.lower()
            or any(search_lower in f.lower() for f in c.overlapping_files)
        ):
            continue
        results.append(c.model_dump())
        if len(results) >= limit > 0:
            break
    return results


@app.get("/api/v1/hotspots")
def get_hotspots(
    subsystem: Optional[str] = Query(None),
    limit: int = 15,
    repo: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    eng = get_engine(repo)
    subsystem_regex = _regex(subsystem)
    results = []
    for h in eng.hotspots:
        if subsystem_regex:
            path_match = subsystem_regex.search(h.path)
            pr_match = any(subsystem_regex.search(eng.pr_map[n].title) for n in h.pr_numbers if n in eng.pr_map)
            if not path_match and not pr_match:
                continue
        row = h.model_dump()
        row["recent_pr_numbers"] = sorted(
            h.pr_numbers,
            key=lambda n: eng.analyses[n].staleness_days if n in eng.analyses else 9999.0,
        )
        results.append(row)
        if len(results) >= limit:
            break
    return results


@app.get("/api/v1/triage")
def get_triage(
    subsystem: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    per_column: int = Query(60, description="Max cards per column, highest risk first"),
    repo: Optional[str] = Query(None),
) -> Dict[str, Any]:
    eng = get_engine(repo)
    subsystem_regex = _regex(subsystem)
    search_lower = search.lower() if search else None

    columns: Dict[str, List[Dict[str, Any]]] = {
        "ready_to_merge": [], "blocked_by_dep": [], "waiting_review": [],
        "waiting_author": [], "failing_ci": [], "draft": [],
    }

    for pr in eng.prs:
        if subsystem_regex and not subsystem_regex.search(pr.title):
            continue
        if search_lower and not (
            search_lower in pr.title.lower() or search_lower in pr.author.lower() or search_lower in str(pr.number)
        ):
            continue

        analysis = eng.analyses.get(pr.number)
        status_val = analysis.triage_status.value if analysis else "WAITING_REVIEW"
        dossier = eng.get_pr_dossier(pr.number)

        if pr.is_draft:
            columns["draft"].append(dossier)
        elif status_val == TriageStatus.READY_TO_MERGE.value:
            columns["ready_to_merge"].append(dossier)
        elif status_val == TriageStatus.BLOCKED_BY_DEPENDENCY.value:
            columns["blocked_by_dep"].append(dossier)
        elif status_val in (TriageStatus.BLOCKED_BY_CI.value, TriageStatus.BLOCKED_BY_CONFLICT.value):
            columns["failing_ci"].append(dossier)
        elif status_val == TriageStatus.WAITING_FOR_AUTHOR.value:
            columns["waiting_author"].append(dossier)
        else:
            columns["waiting_review"].append(dossier)

    totals = {k: len(v) for k, v in columns.items()}
    for k, col in columns.items():
        col.sort(key=lambda d: d["risk_score"], reverse=True)
        columns[k] = col[:per_column] if per_column > 0 else col
    columns["_totals"] = totals
    return columns


@app.get("/api/v1/risk")
def get_risk(
    min_level: Optional[str] = Query(None, description="LOW, MEDIUM, HIGH, CRITICAL"),
    subsystem: Optional[str] = Query(None),
    limit: int = 50,
    repo: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """PR dossiers ranked by change-risk score, highest first."""
    eng = get_engine(repo)
    subsystem_regex = _regex(subsystem)
    rows = eng.get_risk_ranked(min_level)
    if subsystem_regex:
        rows = [d for d in rows if subsystem_regex.search(d["title"])]
    return rows[:limit]


@app.get("/api/v1/reviewers")
def get_reviewers(
    limit: int = 50,
    search: Optional[str] = Query(None),
    repo: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """Per-reviewer workload: pending requests, reviews submitted, oldest pending wait."""
    eng = get_engine(repo)
    rows = eng.reviewer_load
    if search:
        q = search.lower()
        rows = [rl for rl in rows if q in rl.reviewer.lower()]
    return [rl.model_dump() for rl in rows[:limit]]


@app.get("/api/v1/reviewers/{reviewer_handle}")
def get_reviewer_inbox(
    reviewer_handle: str,
    repo: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Actionable Executive Review Inbox for a specific reviewer:
    Partitions pending PRs into Decision-Ready, Stack Unblockers, Quick Wins, and Waiting on Author/Stale.
    """
    eng = get_engine(repo)
    handle_lower = reviewer_handle.strip().lower()
    r_matches = [r for r in eng.reviewer_load if r.reviewer.lower() == handle_lower]
    summary = r_matches[0].model_dump() if r_matches else {
        "reviewer": reviewer_handle,
        "pending_requests": 0,
        "reviews_submitted": 0,
        "approvals": 0,
        "changes_requested": 0,
        "pending_pr_numbers": [],
        "oldest_pending_days": 0.0,
    }

    pending_nums = set(summary.get("pending_pr_numbers", []))
    for pr in eng.prs:
        analysis = eng.analyses.get(pr.number)
        flow = analysis.review_flow if analysis else None
        if flow and any(req.lower() == handle_lower for req in flow.requested_reviewers):
            pending_nums.add(pr.number)

    dossiers = [eng.get_pr_dossier(n) for n in pending_nums if eng.get_pr_dossier(n) is not None]

    ready_decision = []
    stack_unblockers = []
    quick_wins = []
    stale_or_author = []

    for d in dossiers:
        flow = d.get("review_flow") or {}
        status = d.get("status")
        ci = d.get("ci_status")
        is_draft = d.get("is_draft", False)
        blocks = d.get("blocks") or []
        files = d.get("files_changed_count", 0)
        lines = d.get("lines_changed", 0)
        waiting_on = flow.get("waiting_on")

        if len(blocks) > 0:
            stack_unblockers.append(d)
        if (
            waiting_on == "REVIEWER"
            and not is_draft
            and ci in ("SUCCESS", "PENDING", "UNKNOWN")
            and status not in ("CONFLICTING", "CI_FAILURE", "BLOCKED_BY_DEP", "DRAFT")
        ):
            ready_decision.append(d)
        elif (
            waiting_on == "AUTHOR"
            or status in ("CHANGES_REQUESTED", "CI_FAILURE", "CONFLICTING", "DRAFT")
            or is_draft
            or d.get("staleness_days", 0) >= 45
        ):
            stale_or_author.append(d)
        if (
            (files <= 3 or (lines is not None and lines <= 60) or d.get("risk_level") == "LOW")
            and not is_draft
            and ci == "SUCCESS"
            and status not in ("CONFLICTING", "CI_FAILURE")
        ):
            quick_wins.append(d)

    ready_decision.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
    stack_unblockers.sort(key=lambda x: len(x.get("blocks", [])), reverse=True)
    quick_wins.sort(key=lambda x: (x.get("lines_changed") or 0))
    stale_or_author.sort(key=lambda x: x.get("staleness_days", 0), reverse=True)
    dossiers.sort(key=lambda x: x.get("risk_score", 0), reverse=True)

    return {
        "reviewer": summary.get("reviewer", reviewer_handle),
        "summary": summary,
        "counts": {
            "ready_decision": len(ready_decision),
            "stack_unblockers": len(stack_unblockers),
            "quick_wins": len(quick_wins),
            "stale_or_author": len(stale_or_author),
            "all_pending": len(dossiers),
        },
        "buckets": {
            "ready_decision": ready_decision,
            "stack_unblockers": stack_unblockers,
            "quick_wins": quick_wins,
            "stale_or_author": stale_or_author,
            "all_pending": dossiers,
        },
    }


@app.get("/api/v1/prs/{pr_number}")
def get_pr_details(pr_number: int, repo: Optional[str] = Query(None)) -> Dict[str, Any]:
    eng = get_engine(repo)
    dossier = eng.get_pr_dossier(pr_number)
    if not dossier:
        raise HTTPException(status_code=404, detail=f"PR #{pr_number} not found in synced index")
    return dossier


@app.get("/api/v1/simulate/merge/{pr_number}")
def simulate_merge(pr_number: int, repo: Optional[str] = Query(None)) -> Dict[str, Any]:
    """
    Merge Impact Simulator: which downstream PRs are unblocked if this PR lands,
    and which concurrent PRs are at risk of merge conflicts.
    """
    eng = get_engine(repo)
    pr = eng.pr_map.get(pr_number)
    node_data = eng.dag.graph.nodes.get(pr_number, {}) if eng.dag else {}
    if not pr and not node_data:
        raise HTTPException(status_code=404, detail=f"PR #{pr_number} not found in index or graph")

    direct_dependents = eng.dag.get_direct_dependents(pr_number) if eng.dag else []
    unblocked_prs = []
    for dep_num in direct_dependents:
        dep_pr = eng.pr_map.get(dep_num)
        dep_node = eng.dag.graph.nodes.get(dep_num, {})
        other_blockers = [b for b in eng.dag.get_direct_blockers(dep_num) if b != pr_number]
        unblocked_prs.append({
            "pr_number": dep_num,
            "title": dep_pr.title if dep_pr else dep_node.get("title", f"PR #{dep_num}"),
            "author": dep_pr.author if dep_pr else dep_node.get("author", "External"),
            "ci_status": dep_pr.ci_status if dep_pr else "UNKNOWN",
            "fully_unblocked": len(other_blockers) == 0,
            "remaining_blockers": other_blockers,
        })

    collision_risks = []
    for c in eng.collisions:
        if c.pr_a == pr_number or c.pr_b == pr_number:
            other_num = c.pr_b if c.pr_a == pr_number else c.pr_a
            other = eng.pr_map.get(other_num)
            collision_risks.append({
                "pr_number": other_num,
                "title": other.title if other else f"PR #{other_num}",
                "author": other.author if other else "Unknown",
                "risk_level": c.risk_level,
                "shared_churn": c.shared_churn,
                "overlapping_files": c.overlapping_files,
                "is_live_race": c.is_live_race,
            })

    return {
        "merged_pr": {
            "number": pr_number,
            "title": pr.title if pr else node_data.get("title", f"PR #{pr_number}"),
            "author": pr.author if pr else node_data.get("author", "External"),
            "ci_status": pr.ci_status if pr else "UNKNOWN",
        },
        "unblocks_count": len(unblocked_prs),
        "unblocked_prs": unblocked_prs,
        "collision_hazards_count": len(collision_risks),
        "collision_hazards": collision_risks,
    }


@app.post("/api/v1/sync")
def trigger_sync(
    background_tasks: BackgroundTasks,
    limit: int = SYNC_LIMIT,
    force_refresh: bool = False,
    repo: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Refresh from GitHub. Default: synchronous incremental refresh (only PRs
    updated since the last sync). force_refresh=true: full re-fetch in the
    background.
    """
    slug = _resolve_repo(repo)

    def do_sync(full: bool):
        eng = PRRadarEngine(repo_slug=slug)
        eng.sync(limit=limit, use_cache=not full, max_age_minutes=0.0)
        with _lock:
            _engines[slug] = eng
            _refresh_attempts[slug] = datetime.now(timezone.utc)

    if force_refresh:
        background_tasks.add_task(do_sync, True)
        return {"status": "sync_started", "message": f"Full background re-fetch started for {slug}"}

    do_sync(False)
    eng = _engines[slug]
    info = eng.sync_info
    return {
        "status": "success",
        "message": (
            f"{info.get('mode', 'cache')} sync for {slug}: {len(eng.prs)} open PRs indexed, "
            f"{info.get('fetched', 0)} updated, {info.get('evicted', 0)} closed/merged removed"
            + (f" ({info['note']})" if info.get("note") else "")
        ),
        "last_synced_at": eng.last_synced_at.isoformat() if eng.last_synced_at else None,
        "sync_info": info,
    }


# -------------------------------------------------------------
# Web UI
# -------------------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent / "web"


@app.get("/", response_class=HTMLResponse)
@app.get("/overview", response_class=HTMLResponse)
@app.get("/dag", response_class=HTMLResponse)
@app.get("/collisions", response_class=HTMLResponse)
@app.get("/triage", response_class=HTMLResponse)
@app.get("/reviewers", response_class=HTMLResponse)
@app.get("/simulator", response_class=HTMLResponse)
def serve_frontend():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
    <h1>PR Radar API is running!</h1>
    <p>API documentation available at <a href="/docs">/docs</a></p>
    """)
