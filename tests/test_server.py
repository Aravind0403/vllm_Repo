from pathlib import Path

import pytest
from fastapi.testclient import TestClient

CACHE = Path(__file__).resolve().parent.parent / "data" / "vllm-project_vllm_prs.json"
pytestmark = pytest.mark.skipif(not CACHE.exists(), reason="needs data/ cache (run `pr-radar sync` first)")

from pr_radar.server import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vllm" in data["repo"]
    assert data["offline"] is True
    assert data["last_synced_at"] is not None


def test_overview_endpoint():
    response = client.get("/api/v1/overview")
    assert response.status_code == 200
    data = response.json()
    for key in ("total_open_prs", "dependency_stacks_count", "critical_file_collisions",
                "risk_breakdown", "risk_breakdown_actionable", "risk_formula", "top_risk_prs",
                "waiting_on_breakdown", "waiting_on_detail", "top_reviewer_load",
                "top_live_collisions", "top_stack_unblockers"):
        assert key in data
    total = data["total_open_prs"]
    assert sum(data["risk_breakdown"].values()) == total
    assert sum(data["status_breakdown"].values()) == total
    assert sum(data["waiting_on_breakdown"].values()) == total
    assert sum(data["waiting_on_detail"].values()) == total
    # Exact reconciliation invariants
    sb = data["status_breakdown"]
    wb = data["waiting_on_breakdown"]
    wd = data["waiting_on_detail"]
    assert sb.get("READY_TO_MERGE", 0) == wb.get("MERGE", 0) == wd["merge_ready"]
    assert wd["reviewer_active"] + wd["reviewer_stale"] == wb.get("REVIEWER", 0)
    assert wd["author_ci_or_conflict"] + wd["author_feedback"] == wb.get("AUTHOR", 0)
    assert wd["draft"] + wd["stack_blocked"] == wb.get("NONE", 0)
    # Risk formula spec is published
    assert data["risk_formula"]["thresholds"]["CRITICAL"] == 80
    assert len(data["risk_formula"]["path_rules"]) > 0
    # Reviewer stale invariant: actionable_stale_30d <= stale_pending_30d
    for rl in data["top_reviewer_load"]:
        assert rl["actionable_pending"] <= rl["pending_requests"]
        assert rl["actionable_stale_30d"] <= rl["stale_pending_30d"]
        assert rl["is_team"] is False


def test_subsystems_endpoint():
    subsystems = client.get("/api/v1/subsystems").json()
    assert len(subsystems) >= 5
    ids = [s["id"] for s in subsystems]
    assert "V1" in ids and "QUANT" in ids


def test_dag_endpoint():
    data = client.get("/api/v1/dag?scope=stacks").json()
    assert "nodes" in data and "edges" in data and "chains" in data
    for n in data["nodes"]:
        assert "risk_level" in n


def test_collisions_and_hotspots_endpoints():
    assert isinstance(client.get("/api/v1/collisions").json(), list)
    assert isinstance(client.get("/api/v1/hotspots").json(), list)


def test_triage_endpoint_has_all_columns_sorted_by_risk():
    data = client.get("/api/v1/triage").json()
    for col in ["ready_to_merge", "blocked_by_dep", "waiting_review", "waiting_author", "failing_ci", "draft"]:
        assert col in data
        scores = [d["risk_score"] for d in data[col]]
        assert scores == sorted(scores, reverse=True)


def test_risk_endpoint_sorted_and_filtered():
    rows = client.get("/api/v1/risk").json()
    assert rows and all("risk_reasons" in r for r in rows)
    scores = [r["risk_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)

    high = client.get("/api/v1/risk?min_level=HIGH").json()
    assert all(r["risk_level"] in ("HIGH", "CRITICAL") for r in high)


def test_reviewers_endpoint():
    rows = client.get("/api/v1/reviewers").json()
    assert isinstance(rows, list)


def test_pr_dossier_contains_risk_and_review_flow():
    first = client.get("/api/v1/risk?limit=1").json()[0]
    d = client.get(f"/api/v1/prs/{first['number']}").json()
    assert d["risk_level"] == first["risk_level"]
    assert d["review_flow"]["waiting_on"] in ("REVIEWER", "AUTHOR", "MERGE", "NONE")


def test_simulate_merge_endpoint():
    response = client.get("/api/v1/simulate/merge/61779")
    if response.status_code == 200:
        data = response.json()
        assert "unblocked_prs" in data and "collision_hazards" in data
        assert data["merged_pr"]["number"] == 61779


def test_invalid_repo_rejected():
    assert client.get("/api/v1/health?repo=../x").status_code == 400
    assert client.get("/api/v1/health?repo=nope").status_code == 400


def test_sync_offline_reports_cache_mode():
    data = client.post("/api/v1/sync").json()
    assert data["status"] == "success"
    assert data["sync_info"]["mode"] == "cache"


def test_serve_frontend_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "PR Radar" in response.text
    assert "repoSelect" in response.text


def test_dag_subsystem_filter_filters_chains_and_includes_cycles():
    all_dag = client.get("/api/v1/dag?scope=stacks").json()
    rocm_dag = client.get("/api/v1/dag?scope=stacks&subsystem=ROCm").json()
    assert "cycles" in all_dag
    assert len(rocm_dag["chains"]) < len(all_dag["chains"])


def test_invalid_subsystem_regex_returns_400():
    assert client.get("/api/v1/triage?subsystem=[").status_code == 400
    assert client.get("/api/v1/dag?subsystem=[").status_code == 400


@pytest.mark.parametrize("path", ["/", "/overview", "/dag", "/collisions", "/triage", "/reviewers", "/simulator"])
def test_all_spa_page_routes_serve_html(path):
    response = client.get(path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PAGE_ROUTES" in response.text


def test_hotspots_include_recent_pr_numbers_and_filter_collisions():
    hotspots = client.get("/api/v1/hotspots").json()
    assert hotspots
    top = hotspots[0]
    assert "recent_pr_numbers" in top
    assert len(top["recent_pr_numbers"]) == len(top["pr_numbers"])

    filtered = client.get("/api/v1/collisions", params={"file": top["path"], "limit": 20}).json()
    assert filtered
    assert all(top["path"] in c["overlapping_files"] for c in filtered)

    live_only = client.get("/api/v1/collisions", params={"file": top["path"], "live_only": "true", "limit": 20}).json()
    assert all(c["is_live_race"] for c in live_only)

    sample = filtered[0]
    searched = client.get("/api/v1/collisions", params={"file": top["path"], "search": str(sample["pr_a"])}).json()
    assert any(c["pr_a"] == sample["pr_a"] or c["pr_b"] == sample["pr_a"] for c in searched)


def test_reviewers_search_filters_by_handle():
    rows = client.get("/api/v1/reviewers", params={"limit": 10}).json()
    assert isinstance(rows, list)
    if rows:
        handle = rows[0]["reviewer"]
        matched = client.get("/api/v1/reviewers", params={"search": handle[:4]}).json()
        assert any(r["reviewer"] == handle for r in matched)
        assert all(handle[:4].lower() in r["reviewer"].lower() for r in matched)


def test_reviewer_inbox_drilldown():
    res = client.get("/api/v1/reviewers/njhill")
    assert res.status_code == 200
    data = res.json()
    assert data["reviewer"].lower() == "njhill"
    assert "counts" in data
    assert "buckets" in data
    for bucket in ("ready_decision", "stack_unblockers", "quick_wins", "stale_or_author", "all_pending"):
        assert bucket in data["counts"]
        assert bucket in data["buckets"]
        assert isinstance(data["buckets"][bucket], list)
    assert data["counts"]["all_pending"] == len(data["buckets"]["all_pending"])
    # Exact reconciliation between ReviewerLoad.actionable_pending and Inbox ready_decision!
    assert data["counts"]["ready_decision"] == data["summary"]["actionable_pending"]


def test_export_static_site_bundle(tmp_path):
    from pr_radar.engine import PRRadarEngine
    from pr_radar.static_exporter import SPACE_ROUTES, export_static_site

    eng = PRRadarEngine("vllm-project/vllm", offline=True).sync(limit=0)
    summary = export_static_site(eng, tmp_path)
    assert summary["prs"] == len(eng.prs)
    assert (tmp_path / "README.md").exists()
    assert "sdk: static" in (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "window.PR_RADAR_STATIC = true" in (tmp_path / "index.html").read_text(encoding="utf-8")
    assert (tmp_path / "404.html").exists()
    for route in SPACE_ROUTES:
        assert (tmp_path / route / "index.html").exists()
    assert (tmp_path / "vercel.json").exists()
    vercel_cfg = (tmp_path / "vercel.json").read_text(encoding="utf-8")
    assert '"cleanUrls": true' in vercel_cfg
    assert "/index.html" in vercel_cfg
    for name in ("overview.json", "health.json", "subsystems.json", "dag.json", "hotspots.json", "collisions.json", "reviewers.json", "dossiers.json"):
        assert (tmp_path / "static-data" / name).exists()


def test_page_level_reconciliation_invariants():
    """
    Page-level denominator & label contract:
    Every chart on /overview must reconcile to the exact header total_open_prs,
    every status key emitted by the API must be rendered in index.html,
    high-risk header KPI must reconcile to actionable <= all,
    and reviewer coverage must partition On Reviewer exactly.
    """
    health = client.get("/api/v1/health").json()
    overview = client.get("/api/v1/overview").json()
    html = client.get("/overview").text

    total_open_prs = overview["total_open_prs"]
    triage = overview["status_breakdown"]
    moves = overview["waiting_on_breakdown"]
    detail = overview["waiting_on_detail"]
    cov = overview["reviewer_coverage"]

    # 1. Canonical denominator: header == triage sum == moves sum == detail sum
    assert health["prs_indexed"] == total_open_prs
    assert sum(triage.values()) == total_open_prs
    assert sum(moves.values()) == sum(triage.values())
    assert sum(detail.values()) == total_open_prs

    # 2. Every emitted triage status key is explicitly rendered in index.html statusEntries
    for status_key in triage.keys():
        assert f"'{status_key}'" in html, f"Triage status {status_key!r} missing from index.html bars!"

    # 3. High-risk KPI reconciles actionable <= all open
    rb = overview["risk_breakdown"]
    rba = overview["risk_breakdown_actionable"]
    high_all = rb.get("CRITICAL", 0) + rb.get("HIGH", 0)
    high_act = rba.get("CRITICAL", 0) + rba.get("HIGH", 0)
    assert 0 <= high_act <= high_all <= total_open_prs

    # 4. Team/individual reviewer coverage partitions On Reviewer exactly
    assert cov["on_reviewer_total"] == moves.get("REVIEWER", 0)
    assert (
        cov["with_individual_reviewer"] + cov["team_only_reviewer"] + cov["unassigned_reviewer"]
        == moves.get("REVIEWER", 0)
    )

    # 5. Dual stale definitions are explicitly labeled in the UI
    assert "Stale PR (≥14d" in html
    assert "stale req (&gt;30d)" in html or "Stale Req &gt;30d" in html
