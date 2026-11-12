from datetime import datetime, timezone, timedelta
import pytest
from pr_radar.models import DependencyKind, TriageStatus
from pr_radar.dependency_parser import DependencyParser
from pr_radar.collision_detector import CollisionDetector
from pr_radar.dag_engine import DAGEngine
from pr_radar.status_tracker import StatusTracker
from helpers import create_sample_pr


def test_dependency_parser_text_heuristics():
    pr = create_sample_pr(
        number=200,
        body="Stacked on #100 and #105. Also blocked by #110. Fixes bug.",
    )
    rels = DependencyParser.parse_pr(pr, known_pr_numbers={100, 105, 110})
    targets = {r.target_pr: r.kind for r in rels}

    assert 100 in targets
    assert targets[100] == DependencyKind.STACKED_ON
    assert 105 in targets
    assert targets[105] == DependencyKind.STACKED_ON
    assert 110 in targets
    assert targets[110] == DependencyKind.BLOCKED_BY


def test_dependency_parser_branch_base():
    pr = create_sample_pr(
        number=300,
        base_ref="alice/feature-part-1",
        head_ref="alice/feature-part-2",
    )
    branch_map = {"alice/feature-part-1": 299}
    rels = DependencyParser.parse_pr(pr, branch_to_pr=branch_map)

    assert len(rels) == 1
    assert rels[0].target_pr == 299
    assert rels[0].kind == DependencyKind.BRANCH_BASE


def test_collision_detector():
    pr1 = create_sample_pr(1, files=["core/engine.py", "core/utils.py"])
    pr2 = create_sample_pr(2, files=["core/engine.py", "models/llama.py"])
    pr3 = create_sample_pr(3, files=["docs/readme.md"])

    collisions = CollisionDetector.calculate_collisions([pr1, pr2, pr3])
    assert len(collisions) == 1
    c = collisions[0]
    assert c.pr_a == 1 and c.pr_b == 2
    assert c.overlapping_files == ["core/engine.py"]

    hotspots = CollisionDetector.identify_hotspots([pr1, pr2, pr3], min_prs=2)
    assert len(hotspots) == 1
    assert hotspots[0].path == "core/engine.py"
    assert hotspots[0].open_pr_count == 2


def test_dag_engine_and_chains():
    pr1 = create_sample_pr(1, title="Base Foundation")
    pr2 = create_sample_pr(2, title="Middle Layer", body="Depends on #1")
    pr3 = create_sample_pr(3, title="Top Feature", body="Stacked on #2")

    prs = [pr1, pr2, pr3]
    rels = []
    for pr in prs:
        rels.extend(DependencyParser.parse_pr(pr, known_pr_numbers={1, 2, 3}))

    dag = DAGEngine(prs, rels)

    # Foundational unblocker should be PR 1 (has dependents, no blockers)
    foundational = dag.get_foundational_prs()
    assert foundational == [1]

    # Stacks / Chains: [1, 2, 3]
    chains = dag.get_dependency_chains()
    assert len(chains) == 1
    assert chains[0] == [1, 2, 3]

    # Blockers
    assert dag.get_direct_blockers(3) == [2]
    assert dag.get_all_blockers(3) == {1, 2}


def test_status_tracker_diagnosis():
    now = datetime.now(timezone.utc)

    # PR 1: Ready to merge
    pr_ready = create_sample_pr(1, review_decision="APPROVED", ci_status="SUCCESS", mergeable="MERGEABLE")
    
    # PR 2: Blocked by PR 1
    pr_blocked = create_sample_pr(2, body="Depends on #1", review_decision="APPROVED")

    # PR 3: Conflicting
    pr_conflict = create_sample_pr(3, mergeable="CONFLICTING")

    # PR 4: CI failure
    pr_ci_fail = create_sample_pr(4, ci_status="FAILURE")

    # PR 5: Stale
    pr_stale = create_sample_pr(5, updated_at=now - timedelta(days=20))

    prs = [pr_ready, pr_blocked, pr_conflict, pr_ci_fail, pr_stale]
    rels = DependencyParser.parse_pr(pr_blocked, known_pr_numbers={1})
    dag = DAGEngine(prs, rels)

    analyses = {a.pr.number: a for a in StatusTracker.analyze_all(prs, dag, now=now)}

    assert analyses[1].triage_status == TriageStatus.READY_TO_MERGE
    assert analyses[2].triage_status == TriageStatus.BLOCKED_BY_DEPENDENCY
    assert analyses[2].blocked_by == [1]
    assert analyses[3].triage_status == TriageStatus.BLOCKED_BY_CONFLICT
    assert analyses[4].triage_status == TriageStatus.BLOCKED_BY_CI
    assert analyses[5].triage_status == TriageStatus.STALE


def test_dependency_parser_ignores_bare_numbers_negations_and_cross_repo_urls():
    pr = create_sample_pr(
        number=500,
        repo="vllm-project/vllm",
        body=(
            "E2E requires 2 GPUs and requires 16-byte-aligned buffers.\n"
            "Omni parent 670aa7ec, parent 512 MiB, foundation-model-stack #81.\n"
            "No longer blocked by #99.\n"
            "Requires https://github.com/ROCm/aiter/pull/6850 and https://github.com/someone/vllm/pull/1.\n"
            "Depends on #41 and PR 43.\n"
            "Blocked by https://github.com/vllm-project/vllm/pull/42.\n"
        ),
    )
    rels = DependencyParser.parse_pr(pr)
    assert [r.target_pr for r in rels] == [41, 43, 42]


def test_dag_engine_surfaces_cycles():
    pr1 = create_sample_pr(1, body="Blocked by #2")
    pr2 = create_sample_pr(2, body="Blocked by #1")
    rels = DependencyParser.parse_pr(pr1) + DependencyParser.parse_pr(pr2)
    dag = DAGEngine([pr1, pr2], rels)
    assert dag.get_cycles()
    assert dag.get_dependency_chains() == [[1, 2]]

