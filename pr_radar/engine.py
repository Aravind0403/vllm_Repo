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
from .risk_scorer import score_pr, get_risk_formula_spec
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
        self.repo_pulse: Optional[Dict[str, Any]] = None
        self.daily_history: List[Dict[str, Any]] = []

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
        self.repo_pulse = self.fetcher.fetch_repo_pulse(
            self.repo_slug, use_cache=use_cache, max_age_minutes=max_age_minutes
        )
        self._analyze()
        self._record_snapshot()
        return self

    def _record_snapshot(self) -> None:
        if not self.prs:
            return
        now = self.last_synced_at or datetime.now(timezone.utc)
        status_counts: Dict[str, int] = {}
        review_waits: List[float] = []
        for a in self.analyses.values():
            st = a.triage_status.value
            status_counts[st] = status_counts.get(st, 0) + 1
            if a.review_flow and a.review_flow.waiting_on == "REVIEWER":
                review_waits.append(a.review_flow.waiting_days)
        median_wait = sorted(review_waits)[len(review_waits) // 2] if review_waits else 0.0
        snap = {
            "date": now.strftime("%Y-%m-%d"),
            "open_prs": len(self.prs),
            "ready_to_merge": status_counts.get("READY_TO_MERGE", 0),
            "waiting_review": status_counts.get("WAITING_REVIEW", 0),
            "ci_failure": status_counts.get("CI_FAILURE", 0),
            "draft": status_counts.get("DRAFT", 0),
            "median_wait_days": round(median_wait, 1),
        }
        self.daily_history = self.fetcher.record_daily_snapshot(self.repo_slug, snap)

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

        # 6. Risk score + 7. Review flow (strictly reconciled with StatusTracker)
        flows = {}
        for num, analysis in self.analyses.items():
            analysis.risk = score_pr(analysis.pr, collisions_by_pr.get(num, []), self.repo_slug)
            flow = analyze_review_flow(analysis.pr, now, blocked_by=analysis.blocked_by)
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

        self.reviewer_load = compute_reviewer_load(self.prs, flows, analyses=self.analyses, now=now)

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
            "risk_components": {
                "sensitivity": risk.sensitivity if risk else 0.0,
                "churn": risk.churn if risk else 0.0,
                "collision": risk.collision if risk else 0.0,
            },
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
        """High-level repository health and triage stats, strictly reconciled."""
        status_counts: Dict[str, int] = {}
        risk_counts: Dict[str, int] = {}
        risk_counts_actionable: Dict[str, int] = {}
        waiting_counts: Dict[str, int] = {}
        waiting_detail: Dict[str, int] = {
            "reviewer_active": 0,
            "reviewer_stale": 0,
            "author_ci_or_conflict": 0,
            "author_feedback": 0,
            "merge_ready": 0,
            "draft": 0,
            "stack_blocked": 0,
        }
        reviewer_coverage: Dict[str, int] = {
            "on_reviewer_total": 0,
            "with_individual_reviewer": 0,
            "team_only_reviewer": 0,
            "unassigned_reviewer": 0,
        }

        for a in self.analyses.values():
            st = a.triage_status.value
            status_counts[st] = status_counts.get(st, 0) + 1
            w = a.review_flow.waiting_on if a.review_flow else "REVIEWER"
            waiting_counts[w] = waiting_counts.get(w, 0) + 1

            if w == "REVIEWER":
                reviewer_coverage["on_reviewer_total"] += 1
                has_ind = bool(a.pr.requested_reviewers)
                has_team = bool(a.pr.requested_teams)
                if has_ind:
                    reviewer_coverage["with_individual_reviewer"] += 1
                elif has_team:
                    reviewer_coverage["team_only_reviewer"] += 1
                else:
                    reviewer_coverage["unassigned_reviewer"] += 1

            if a.risk:
                risk_counts[a.risk.level] = risk_counts.get(a.risk.level, 0) + 1
                if w in ("REVIEWER", "MERGE"):
                    risk_counts_actionable[a.risk.level] = risk_counts_actionable.get(a.risk.level, 0) + 1

            if st == "READY_TO_MERGE":
                waiting_detail["merge_ready"] += 1
            elif st == "DRAFT":
                waiting_detail["draft"] += 1
            elif st == "BLOCKED_BY_DEP":
                waiting_detail["stack_blocked"] += 1
            elif st in ("CI_FAILURE", "CONFLICTING"):
                waiting_detail["author_ci_or_conflict"] += 1
            elif st == "CHANGES_REQUESTED":
                waiting_detail["author_feedback"] += 1
            elif st == "STALE":
                if w == "REVIEWER":
                    waiting_detail["reviewer_stale"] += 1
                else:
                    waiting_detail["author_feedback"] += 1
            else:
                if w == "REVIEWER":
                    waiting_detail["reviewer_active"] += 1
                else:
                    waiting_detail["author_feedback"] += 1

        foundational = self.dag.get_foundational_prs() if self.dag else []
        chains = self.dag.get_dependency_chains() if self.dag else []
        critical_collisions = [c for c in self.collisions if c.risk_level == "CRITICAL"]
        live_races = [c for c in self.collisions if c.is_live_race and c.risk_level in ("CRITICAL", "HIGH")]
        live_races.sort(key=lambda c: (c.risk_level != "CRITICAL", -c.collision_score, -c.shared_churn))

        # Curate diverse, non-stacked top 5 live merge races for the Overview spotlight
        same_stack_pairs: Set[tuple] = set()
        for ch in chains:
            for i in range(len(ch)):
                for j in range(i + 1, len(ch)):
                    same_stack_pairs.add((min(ch[i], ch[j]), max(ch[i], ch[j])))
        curated_live_races: List[FileCollision] = []
        seen_race_prs: Set[int] = set()
        for c in live_races:
            pair = (min(c.pr_a, c.pr_b), max(c.pr_a, c.pr_b))
            if pair in same_stack_pairs:
                continue
            if c.pr_a in seen_race_prs or c.pr_b in seen_race_prs:
                continue
            curated_live_races.append(c)
            seen_race_prs.add(c.pr_a)
            seen_race_prs.add(c.pr_b)
            if len(curated_live_races) >= 5:
                break
        if len(curated_live_races) < 5:
            for c in live_races:
                if c not in curated_live_races:
                    curated_live_races.append(c)
                if len(curated_live_races) >= 5:
                    break

        # Top foundational stack unblockers with fan-out & chain preview
        unblocker_rows: List[Dict[str, Any]] = []
        if self.dag:
            chains_by_base: Dict[int, List[List[int]]] = {}
            for ch in chains:
                if ch:
                    chains_by_base.setdefault(ch[0], []).append(ch)
            for base_num in foundational:
                if base_num not in self.analyses:
                    continue
                a = self.analyses[base_num]
                downstream_all = self.dag.get_all_downstream_dependents(base_num) if hasattr(self.dag, "get_all_downstream_dependents") else set(a.blocks)
                # Fallback transitive count via chains if method doesn't exist
                if not hasattr(self.dag, "get_all_downstream_dependents"):
                    downstream_set: Set[int] = set(a.blocks)
                    for ch in chains_by_base.get(base_num, []):
                        downstream_set.update(ch[1:])
                    downstream_all = downstream_set
                longest_chain = max(chains_by_base.get(base_num, [[base_num]]), key=len)
                unblocker_rows.append({
                    "number": base_num,
                    "title": a.pr.title,
                    "author": a.pr.author,
                    "url": a.pr.url,
                    "status": a.triage_status.value,
                    "ci_status": a.pr.ci_status,
                    "risk_level": a.risk.level if a.risk else "LOW",
                    "risk_score": a.risk.score if a.risk else 0,
                    "direct_dependents_count": len(a.blocks),
                    "downstream_total_count": len(downstream_all),
                    "chain_preview": longest_chain[:5],
                    "chain_length": len(longest_chain),
                })
            unblocker_rows.sort(
                key=lambda r: (
                    r["downstream_total_count"],
                    r["status"] == "READY_TO_MERGE",
                    r["ci_status"] == "SUCCESS",
                    r["chain_length"],
                ),
                reverse=True,
            )

        # Top actionable risk PRs first (fallback to all if none)
        actionable_risk = [
            a for a in self.analyses.values()
            if a.risk and a.review_flow and a.review_flow.waiting_on in ("REVIEWER", "MERGE")
        ]
        risk_pool = actionable_risk if actionable_risk else [a for a in self.analyses.values() if a.risk]
        top_risk = sorted(risk_pool, key=lambda a: a.risk.score, reverse=True)[:5]

        review_waits = [a.review_flow.waiting_days for a in self.analyses.values()
                        if a.review_flow and a.review_flow.waiting_on == "REVIEWER"]
        median_wait = sorted(review_waits)[len(review_waits) // 2] if review_waits else 0.0

        individual_reviewers = [rl for rl in self.reviewer_load if not rl.is_team]

        pulse = dict(self.repo_pulse) if self.repo_pulse else None
        if pulse is not None:
            pulse["daily_snapshots"] = self.daily_history

        return {
            "repo": self.repo_slug,
            "total_open_prs_analyzed": len(self.prs),
            "last_synced_at": self.last_synced_at.isoformat() if self.last_synced_at else None,
            "sync_mode": self.sync_info.get("mode"),
            "status_breakdown": status_counts,
            "foundational_unblockers": foundational,
            "top_stack_unblockers": unblocker_rows[:5],
            "dependency_stacks_count": len(chains),
            "critical_file_collisions": len(critical_collisions),
            "live_race_collisions_count": len(live_races),
            "top_live_collisions": [c.model_dump() for c in curated_live_races],
            "top_hotspot_files": [
                {"path": h.path, "pr_count": h.open_pr_count, "churn": h.total_churn, "active_prs_count": h.active_prs_count}
                for h in self.hotspots[:5]
            ],
            "risk_breakdown": risk_counts,
            "risk_breakdown_actionable": risk_counts_actionable,
            "risk_formula": get_risk_formula_spec(self.repo_slug),
            "top_risk_prs": [
                {
                    "number": a.pr.number,
                    "title": a.pr.title,
                    "status": a.triage_status.value,
                    "risk_score": a.risk.score,
                    "risk_level": a.risk.level,
                    "sensitive_areas": a.risk.sensitive_areas,
                    "risk_components": {
                        "sensitivity": a.risk.sensitivity,
                        "churn": a.risk.churn,
                        "collision": a.risk.collision,
                    },
                }
                for a in top_risk
            ],
            "waiting_on_breakdown": waiting_counts,
            "waiting_on_detail": waiting_detail,
            "reviewer_coverage": reviewer_coverage,
            "median_reviewer_wait_days": round(median_wait, 1),
            "top_reviewer_load": [
                rl.model_dump() for rl in individual_reviewers[:5]
            ],
            "repo_pulse": pulse,
        }
