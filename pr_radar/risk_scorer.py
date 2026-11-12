from __future__ import annotations

import re
from typing import List, Tuple, Dict, Optional
from .models import PullRequest, FileCollision, RiskScore


# (regex over file path, weight 0..1, human label)
PathRule = Tuple[str, float, str]

VLLM_SENSITIVE_PATHS: List[PathRule] = [
    (r"^vllm/v1/core/sched/", 1.0, "V1 scheduler"),
    (r"^vllm/v1/core/(kv_cache|block_pool|single_type_kv_cache)", 1.0, "KV cache / memory"),
    (r"^vllm/v1/kv_cache_interface\.py", 1.0, "KV cache / memory"),
    (r"^vllm/v1/worker/", 0.8, "V1 worker / model runner"),
    (r"^csrc/", 0.9, "CUDA / C++ kernels"),
    (r"^vllm/(v1/)?attention/", 0.8, "Attention backends"),
    (r"^vllm/model_executor/layers/attention/", 0.8, "Attention backends"),
    (r"^vllm/model_executor/layers/quantization/", 0.7, "Quantization"),
    (r"^vllm/model_executor/layers/fused_moe/", 0.7, "Fused MoE kernels"),
    (r"^vllm/model_executor/layers/mamba/", 0.6, "Mamba / linear attention"),
    (r"^vllm/platforms/", 0.7, "Hardware platforms"),
    (r"^vllm/distributed/", 0.7, "Distributed / parallelism"),
    (r"^vllm/(v1/)?engine/", 0.6, "Engine core"),
    (r"^vllm/v1/spec_decode/", 0.6, "Speculative decoding"),
    (r"^vllm/compilation/", 0.6, "torch.compile / CUDA graphs"),
    (r"^vllm/entrypoints/", 0.4, "API & serving"),
    (r"^vllm/model_executor/models/", 0.3, "Model definitions"),
    (r"^vllm/config/", 0.4, "Configuration"),
    (r"^(setup\.py|pyproject\.toml|CMakeLists\.txt|requirements.*\.txt)$", 0.5, "Build / packaging"),
    (r"^tests/", 0.1, "Tests"),
    (r"^(docs/|.*\.md$|.*\.rst$)", 0.0, "Documentation"),
]

GENERIC_SENSITIVE_PATHS: List[PathRule] = [
    (r"sched", 0.9, "Scheduler"),
    (r"kv_?cache|memory_?pool|block_?pool|allocator", 0.9, "KV cache / memory"),
    (r"csrc/|\.cu$|\.cuh$|kernel", 0.8, "Kernels"),
    (r"attention|attn", 0.7, "Attention backends"),
    (r"quant", 0.6, "Quantization"),
    (r"moe", 0.6, "MoE"),
    (r"distributed|parallel", 0.6, "Distributed / parallelism"),
    (r"engine|runtime|server_args|environ", 0.5, "Engine core"),
    (r"^tests?/|_test\.py$|test_", 0.1, "Tests"),
    (r"^docs?/|\.mdx?$|\.rst$", 0.0, "Documentation"),
]

REPO_PATH_TABLES: Dict[str, List[PathRule]] = {
    "vllm-project/vllm": VLLM_SENSITIVE_PATHS,
}

CHURN_SATURATION_LINES = 1500
WEIGHT_SENSITIVITY = 0.5
WEIGHT_CHURN = 0.2
WEIGHT_COLLISION = 0.3


def path_table_for(repo_slug: str) -> List[PathRule]:
    return REPO_PATH_TABLES.get(repo_slug, GENERIC_SENSITIVE_PATHS)


def _level(score: int) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 30:
        return "MEDIUM"
    return "LOW"


def score_pr(
    pr: PullRequest,
    collisions: Optional[List[FileCollision]] = None,
    repo_slug: Optional[str] = None,
) -> RiskScore:
    """
    Explainable risk score for one PR.

    sensitivity: max weight of any matched sensitive path, boosted by how many
                 sensitive files are touched (capped at 1.0)
    churn:       total changed lines / CHURN_SATURATION_LINES (capped at 1.0)
    collision:   0.5 per CRITICAL + 0.25 per HIGH collision (capped at 1.0)
    score:       100 * (0.5 * sensitivity + 0.2 * churn + 0.3 * collision)
    """
    table = [(re.compile(rx), w, label) for rx, w, label in path_table_for(repo_slug or pr.repo)]
    reasons: List[str] = []

    # --- sensitivity
    area_hits: Dict[str, Tuple[float, int]] = {}   # label -> (weight, file count)
    for f in pr.files:
        for regex, weight, label in table:
            if regex.search(f.path):
                w, n = area_hits.get(label, (weight, 0))
                area_hits[label] = (max(w, weight), n + 1)
                break  # first matching rule wins (table is ordered specific -> generic)

    sensitive = {label: (w, n) for label, (w, n) in area_hits.items() if w >= 0.5}
    if sensitive:
        max_w = max(w for w, _ in sensitive.values())
        n_files = sum(n for _, n in sensitive.values())
        sensitivity = min(1.0, max_w * (1.0 + 0.05 * (n_files - 1)))
        for label, (w, n) in sorted(sensitive.items(), key=lambda kv: -kv[1][0]):
            reasons.append(f"Touches {label} ({n} file{'s' if n != 1 else ''}, weight {w:.1f})")
    else:
        sensitivity = max((w for w, _ in area_hits.values()), default=0.0)
        if pr.files and sensitivity == 0.0:
            reasons.append("Only documentation or non-sensitive paths touched")

    # --- churn
    total = pr.total_changes
    churn = min(1.0, total / CHURN_SATURATION_LINES) if total else 0.0
    if total >= 1000:
        reasons.append(f"Very large change: {total:,} lines")
    elif total >= 300:
        reasons.append(f"Large change: {total:,} lines")

    # --- collisions: only live races count (both sides pushed within 5 days).
    # A stale overlap with an abandoned PR is that PR's problem, not this one's.
    live = [c for c in (collisions or []) if c.is_live_race]
    critical = [c for c in live if c.risk_level == "CRITICAL"]
    high = [c for c in live if c.risk_level == "HIGH"]
    collision = min(1.0, 0.4 * len(critical) + 0.2 * len(high))
    if critical:
        others = ", ".join(f"#{c.pr_b if c.pr_a == pr.number else c.pr_a}" for c in critical[:3])
        reasons.append(f"{len(critical)} live CRITICAL collision{'s' if len(critical) != 1 else ''} with {others} (both active)")
    if high:
        reasons.append(f"{len(high)} live HIGH collision{'s' if len(high) != 1 else ''}")
    stale = len((collisions or [])) - len(live)
    if stale and (critical or high or stale >= 5):
        reasons.append(f"{stale} stale overlap{'s' if stale != 1 else ''} with inactive PRs (not scored)")

    score = int(round(100 * (
        WEIGHT_SENSITIVITY * sensitivity
        + WEIGHT_CHURN * churn
        + WEIGHT_COLLISION * collision
    )))
    score = max(0, min(100, score))

    if pr.ci_status == "FAILURE" and score >= 25:
        reasons.append("CI is failing on a sensitive change")

    return RiskScore(
        score=score,
        level=_level(score),
        reasons=reasons,
        sensitive_areas=sorted(sensitive.keys(), key=lambda k: -sensitive[k][0]),
        sensitivity=round(sensitivity, 3),
        churn=round(churn, 3),
        collision=round(collision, 3),
    )
