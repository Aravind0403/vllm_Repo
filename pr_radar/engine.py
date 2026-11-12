from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Optional, Set, Any
from .models import (
    PullRequest,
    DependencyRelation,
    FileCollision,
    HotspotFile,
    PRAnalysis,
    TriageStatus,
    ReviewerLoad,
)
from .fetcher import GitHubFetcher
from .dependency_parser import DependencyParser
from .collision_detector import CollisionDetector
from .dag_engine import DAGEngine
from .status_tracker import StatusTracker
from .risk_scorer import score_pr
from .review_flow import analyze_review_flow, compute_reviewer_load


class PRRadarEngine:
    """
    Unified Orchestrator:
    Loads PRs -> Parses Dependencies -> Builds DAG -> Computes Collisions
    -> Triages Status -> Scores Risk -> Analyzes Review Flow.
    """

    def __init__(self, repo_slug: str, cache_dir: str = "data", offline: Optional[bool] = None):
        self.repo_slug = repo_slug
        self.cache_dir = cache_dir
        self.fetcher = GitHubFetcher(cache_dir=cache_dir, offline=offline)

        self.prs: List[PullRequest] = []
        self.pr_map: Dict[int, PullRequest] = {}
        self.relations: List[DependencyRelation] = []
        self.collisions: List[FileCollision] = []
        self.hotspots: List[HotspotFile] = []
        self.dag: Optional[DAGEngine] = None
        self.analyses: Dict[int, PRAnalysis] = {}
        self.reviewer_load: List[ReviewerLoad] = []
        self.collisions_by_pr: Dict[int, List[FileCollision]] = {}
        self.last_synced_at: Optional[datetime] = None
        self.last_analyzed_at: Optional[datetime] = None
        self.sync_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------ sync
    def sync(
        self,
        limit: int = 500,
        use_cache: bool = True,
        max_age_minutes: Optional[float] = 30.0,
    ) -> PRRadarEngine:
        """
        Fetch PRs (cache / incremental / full, see GitHubFetcher) and run the
        full analysis. max_age_minutes=None accepts any present cache as-is.
        """
        self.prs = self.fetcher.fetch_open_prs(
            self.repo_slug, limit=limit, use_cache=use_cache, max_age_minutes=max_age_minutes
        )
        self.last_synced_at = self.fetcher.last_synced_at
        self.sync_info = dict(self.fetcher.last_sync_info)
        self._analyze()
        return self

    def _analyze(self) -> None:
        now = datetime.now(timezone.utc)
        self.last_analyzed_at = now
        self.pr_map = {pr.number: pr for pr in self.prs}

        branch_to_pr: Dict[str, int] = {
            pr.head_ref: pr.number for pr in self.prs if pr.head_ref
        }

        # 1. Parse dependencies
        self.relations = []
        target_candidates: Set[int] = set()
        for pr in self.prs:
            rels = DependencyParser.parse_pr(pr, known_pr_numbers=None, branch_to_pr=branch_to_pr)
            self.relations.extend(rels)
            for r in rels:
                if r.target_pr not in self.pr_map:
                    target_candidates.add(r.target_pr)

        # 2. Resolve external targets (open / merged / closed)
        resolved_targets: Dict[int, Dict[str, Any]] = {}
        if target_candidates:
            resolved_targets = self.fetcher.resolve_target_prs(self.repo_slug, list(target_candidates))

        open_pr_numbers: Set[int] = {pr.number for pr in self.prs if pr.state == "OPEN"}
        for target_num, info in resolved_targets.items():
            if info.get("state") == "OPEN":
                open_pr_numbers.add(target_num)

        # 3. DAG
        self.dag = DAGEngine(self.prs, self.relations)
        for target_num, info in resolved_targets.items():
            if self.dag.graph.has_node(target_num):
                self.dag.graph.nodes[target_num].update({
                    "title": info.get("title", f"PR #{target_num}"),
                    "state": info.get("state", "UNKNOWN"),
                    "is_draft": info.get("isDraft", False),
                    "is_external": True,
                })

        # 4. Collisions & hotspots
        self.collisions = CollisionDetector.calculate_collisions(self.prs)
        self.hotspots = CollisionDetector.identify_hotspots(self.prs)

        # 5. Triage
        analysis_list = [StatusTracker.analyze_pr(pr, self.dag, open_pr_numbers, now) for pr in self.prs]
        self.analyses = {a.pr.number: a for a in analysis_list}

        # Enrich collisions with titles / idle days / live-race flag
        for col in self.collisions:
            a = self.analyses.get(col.pr_a)
            b = self.analyses.get(col.pr_b)
            if a:
                col.pr_a_title, col.pr_a_author, col.pr_a_idle_days = a.pr.title, a.pr.author, a.staleness_days
            if b:
                col.pr_b_title, col.pr_b_author, col.pr_b_idle_days = b.pr.title, b.pr.author, b.staleness_days
            col.is_live_race = col.pr_a_idle_days < 5.0 and col.pr_b_idle_days < 5.0

        for h in self.hotspots:
            idles = [self.analyses[n].staleness_days for n in h.pr_numbers if n in self.analyses]
            if idles:
                h.min_idle_days = min(idles)
                h.active_prs_count = sum(1 for d in idles if d < 5.0)

        collisions_by_pr: Dict[int, List[FileCollision]] = {}
        for col in self.collisions:
            collisions_by_pr.setdefault(col.pr_a, []).append(col)
            collisions_by_pr.setdefault(col.pr_b, []).append(col)
            if col.risk_level in ("CRITICAL", "HIGH"):
                if col.pr_a in self.analyses:
                    self.analyses[col.pr_a].collision_prs.append(col.pr_b)
                if col.pr_b in self.analyses:
                    self.analyses[col.pr_b].collision_prs.append(col.pr_a)
        self.collisions_by_pr = collisions_by_pr

        # 6. Risk score + 7. Review flow
        flows = {}
        for num, analysis in self.analyses.items():
            analysis.risk = score_pr(analysis.pr, collisions_by_pr.get(num, []), self.repo_slug)
            flow = analyze_review_flow(analysis.pr, now)
            analysis.review_flow = flow
            flows[num] = flow

            # Review flow refines the generic "waiting for review" diagnosis
            if (
                analysis.triage_status == TriageStatus.WAITING_FOR_REVIEW
                and flow.has_review_signals
                and flow.waiting_on == "AUTHOR"
            ):
                analysis.triage_status = TriageStatus.WAITING_FOR_AUTHOR
                who = f" from {flow.last_reviewer}" if flow.last_reviewer else ""
                analysis.primary_blocker_reason = (
                    f"Waiting on author: review feedback{who} outstanding for {flow.waiting_days:.0f}d"
                )
            elif (
                analysis.triage_status == TriageStatus.WAITING_FOR_REVIEW
                and flow.has_review_signals
                and flow.waiting_on == "REVIEWER"
            ):
                if flow.requested_reviewers:
                    names = ", ".join(flow.requested_reviewers[:3])
                    analysis.primary_blocker_reason = (
                        f"Waiting on reviewer(s) {names} for {flow.waiting_days:.0f}d"
                    )
                elif flow.review_count == 0:
                    analysis.primary_blocker_reason = (
                        f"No review yet after {flow.waiting_days:.0f}d; no reviewer requested"
                    )
                else:
                    analysis.primary_blocker_reason = (
                        f"Author responded; awaiting re-review for {flow.waiting_days:.0f}d"
                    )

        self.reviewer_load = compute_reviewer_load(self.prs, flows)

    # --------------------------------------------------------------- queries
    def get_pr_dossier(self, pr_number: int) -> Optional[Dict[str, Any]]:
        """Complete deep-dive inspection of a single PR."""
        analysis = self.analyses.get(pr_number)
        if not analysis:
            return None

        pr = analysis.pr
        relevant = self.collisions_by_pr.get(pr_number, [])  # already sorted by risk
        risk = analysis.risk
        flow = analysis.review_flow

        return {
            "number": pr.number,
            "title": pr.title,
            "url": pr.url,
            "author": pr.author,
            "status": analysis.triage_status.value,
            "reason": analysis.primary_blocker_reason,
            "ci_status": pr.ci_status,
            "review_decision": pr.review_decision or "NONE",
            "mergeable": pr.mergeable,
            "is_draft": pr.is_draft,
            "staleness_days": analysis.staleness_days,
            "updated_at_str": pr.updated_at.strftime("%b %d, %Y"),
            "created_at_str": pr.created_at.strftime("%b %d, %Y"),
            "blocked_by": analysis.blocked_by,
            "blocks": analysis.blocks,
            "files_changed_count": len(pr.files),
            "lines_changed": pr.total_changes,
            "collisions_count": len(relevant),
            "live_race_count": sum(1 for c in relevant if c.is_live_race and c.risk_level in ("CRITICAL", "HIGH")),
            "collisions": [
                {
                    "colliding_pr": c.pr_b if c.pr_a == pr_number else c.pr_a,
                    "risk_level": c.risk_level,
                    "collision_score": c.collision_score,
                    "overlapping_files": c.overlapping_files,
                    "is_live_race": c.is_live_race,
                }
                for c in sorted(relevant, key=lambda c: (not c.is_live_race, c.risk_level != "CRITICAL", -c.collision_score))[:5]
            ],
            "risk_score": risk.score if risk else 0,
            "risk_level": risk.level if risk else "LOW",
            "risk_reasons": risk.reasons if risk else [],
            "sensitive_areas": risk.sensitive_areas if risk else [],
            "review_flow": flow.model_dump(mode="json") if flow else None,
        }

    def get_risk_ranked(self, min_level: Optional[str] = None) -> List[Dict[str, Any]]:
        order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
        floor = order.get((min_level or "").upper(), 0)
        rows = []
        for a in self.analyses.values():
            if a.risk and order.get(a.risk.level, 0) >= floor:
                rows.append(self.get_pr_dossier(a.pr.number))
        rows.sort(key=lambda d: d["risk_score"], reverse=True)
        return rows

    def get_summary_stats(self) -> Dict[str, Any]:
        """High-level repository health and triage stats."""
        status_counts: Dict[str, int] = {}
        risk_counts: Dict[str, int] = {}
        waiting_counts: Dict[str, int] = {}
        for a in self.analyses.values():
            st = a.triage_status.value
            status_counts[st] = status_counts.get(st, 0) + 1
            if a.risk:
                risk_counts[a.risk.level] = risk_counts.get(a.risk.level, 0) + 1
            if a.review_flow:
                w = a.review_flow.waiting_on
                waiting_counts[w] = waiting_counts.get(w, 0) + 1

        foundational = self.dag.get_foundational_prs() if self.dag else []
        chains = self.dag.get_dependency_chains() if self.dag else []
        critical_collisions = [c for c in self.collisions if c.risk_level == "CRITICAL"]

        top_risk = sorted(
            (a for a in self.analyses.values() if a.risk),
            key=lambda a: a.risk.score,
            reverse=True,
        )[:5]

        review_waits = [a.review_flow.waiting_days for a in self.analyses.values()
                        if a.review_flow and a.review_flow.waiting_on == "REVIEWER"]
        median_wait = sorted(review_waits)[len(review_waits) // 2] if review_waits else 0.0

        return {
            "repo": self.repo_slug,
            "total_open_prs_analyzed": len(self.prs),
            "last_synced_at": self.last_synced_at.isoformat() if self.last_synced_at else None,
            "sync_mode": self.sync_info.get("mode"),
            "status_breakdown": status_counts,
            "foundational_unblockers": foundational,
            "dependency_stacks_count": len(chains),
            "critical_file_collisions": len(critical_collisions),
            "top_hotspot_files": [
                {"path": h.path, "pr_count": h.open_pr_count, "churn": h.total_churn}
                for h in self.hotspots[:5]
            ],
            "risk_breakdown": risk_counts,
            "top_risk_prs": [
                {"number": a.pr.number, "title": a.pr.title, "risk_score": a.risk.score,
                 "risk_level": a.risk.level, "sensitive_areas": a.risk.sensitive_areas}
                for a in top_risk
            ],
            "waiting_on_breakdown": waiting_counts,
            "median_reviewer_wait_days": round(median_wait, 1),
            "top_reviewer_load": [
                rl.model_dump() for rl in self.reviewer_load[:5]
            ],
        }
