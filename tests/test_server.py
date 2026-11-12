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
                "risk_breakdown", "top_risk_prs", "waiting_on_breakdown", "top_reviewer_load"):
        assert key in data
    assert sum(data["risk_breakdown"].values()) == data["total_open_prs"]


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
