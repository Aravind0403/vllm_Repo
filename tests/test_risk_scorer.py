from pr_radar.risk_scorer import score_pr, path_table_for, VLLM_SENSITIVE_PATHS, GENERIC_SENSITIVE_PATHS
from pr_radar.models import FileCollision
from helpers import create_sample_pr

VLLM = "vllm-project/vllm"


def test_docs_only_pr_is_low_risk():
    pr = create_sample_pr(1, repo=VLLM, files=["docs/source/index.md", "README.md"])
    r = score_pr(pr, [], VLLM)
    assert r.level == "LOW" and r.score < 10
    assert r.sensitive_areas == []
    assert any("documentation" in reason.lower() for reason in r.reasons)


def test_scheduler_and_kv_cache_change_is_high_or_critical():
    pr = create_sample_pr(
        2, repo=VLLM,
        files=["vllm/v1/core/sched/scheduler.py", "vllm/v1/core/kv_cache_manager.py", "tests/v1/test_sched.py"],
        additions=300, deletions=100,
    )
    r = score_pr(pr, [], VLLM)
    assert r.level in ("HIGH", "CRITICAL")
    assert "V1 scheduler" in r.sensitive_areas and "KV cache / memory" in r.sensitive_areas
    assert "Tests" not in r.sensitive_areas
    assert any("V1 scheduler" in reason for reason in r.reasons)


def test_collisions_raise_the_score():
    pr = create_sample_pr(3, repo=VLLM, files=["vllm/attention/backends/flash.py"])
    base = score_pr(pr, [], VLLM)
    col = FileCollision(pr_a=3, pr_b=99, overlapping_files=["vllm/attention/backends/flash.py"],
                        collision_score=0.5, risk_level="CRITICAL", is_live_race=True)
    with_col = score_pr(pr, [col], VLLM)
    assert with_col.score > base.score
    assert with_col.collision == 0.4
    assert any("#99" in reason for reason in with_col.reasons)
    assert any("both active" in reason for reason in with_col.reasons)


def test_churn_component_saturates():
    pr = create_sample_pr(4, repo=VLLM, files=["vllm/model_executor/models/foo.py"], additions=5000, deletions=0)
    r = score_pr(pr, [], VLLM)
    assert r.churn == 1.0
    assert any("Very large change" in reason for reason in r.reasons)


def test_generic_table_used_for_other_repos():
    assert path_table_for(VLLM) is VLLM_SENSITIVE_PATHS
    assert path_table_for("example/other-repo") is GENERIC_SENSITIVE_PATHS
    pr = create_sample_pr(5, repo="example/other-repo", files=["python/engine/managers/scheduler.py"])
    r = score_pr(pr, [])
    assert "Scheduler" in r.sensitive_areas and r.level in ("MEDIUM", "HIGH", "CRITICAL")


def test_score_bounds_and_level_mapping():
    pr = create_sample_pr(6, repo=VLLM, files=["csrc/attention/attn.cu"] * 1, additions=2000, deletions=2000)
    cols = [FileCollision(pr_a=6, pr_b=n, overlapping_files=["csrc/attention/attn.cu"], collision_score=0.9,
                          risk_level="CRITICAL", is_live_race=True) for n in (1, 2, 3)]
    r = score_pr(pr, cols, VLLM)
    assert 0 <= r.score <= 100 and r.level == "CRITICAL"


def test_stale_collisions_do_not_raise_score():
    pr = create_sample_pr(7, repo=VLLM, files=["vllm/attention/backends/flash.py"])
    stale = FileCollision(pr_a=7, pr_b=42, overlapping_files=["vllm/attention/backends/flash.py"],
                          collision_score=0.9, risk_level="CRITICAL", is_live_race=False)
    r = score_pr(pr, [stale], VLLM)
    assert r.collision == 0.0
    assert r.score == score_pr(pr, [], VLLM).score
