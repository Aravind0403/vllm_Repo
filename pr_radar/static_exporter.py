from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

from .engine import PRRadarEngine
from .server import _serialize_dag_node


SUBSYSTEM_DEFINITIONS: List[Dict[str, str]] = [
    {"id": "V1", "name": "V1 Engine", "pattern": r"(?i)v1"},
    {"id": "ATTN", "name": "Attention & Kernels", "pattern": r"(?i)attention|kernel|flash|gdn|triton"},
    {"id": "ROCM", "name": "ROCm / AMD", "pattern": r"(?i)rocm|amd"},
    {"id": "HARDWARE", "name": "TPU & Hardware", "pattern": r"(?i)tpu|neuron|xpu|cuda"},
    {"id": "QUANT", "name": "Quantization", "pattern": r"(?i)quant|fp8|mxfp4|awq|gptq|quark"},
    {"id": "SPEC", "name": "Speculative Decoding", "pattern": r"(?i)spec|eagle|mtp|draft"},
    {"id": "DISTRIBUTED", "name": "Distributed / MoE", "pattern": r"(?i)moe|parallel|distributed"},
    {"id": "API", "name": "API & Serving", "pattern": r"(?i)serve|api|entrypoint|openai"},
]

SPACE_ROUTES = ("overview", "dag", "collisions", "triage", "reviewers", "simulator")


def _space_readme(repo_slug: str, pr_count: int, stacks_count: int) -> str:
    return (
        "---\n"
        "title: vLLM PR Radar\n"
        "emoji: 🛰️\n"
        "colorFrom: green\n"
        "colorTo: indigo\n"
        "sdk: static\n"
        "pinned: false\n"
        "app_file: index.html\n"
        "short_description: Live PR dependency DAG, collision radar & triage for vLLM\n"
        "---\n\n"
        f"# PR Radar — `{repo_slug}`\n\n"
        f"Static snapshot built from **{pr_count:,} open PRs** and **{stacks_count:,} dependency stacks**.\n\n"
        "- `/overview` — Risk summary, triage breakdown, reviewer queues\n"
        "- `/dag` — Dependency stacks and merge order\n"
        "- `/collisions` — Contended hotspot files and live merge races\n"
        "- `/triage` — 6-column triage Kanban board\n"
        "- `/reviewers` — Reviewer pending queue & unanswered wait time\n"
        "- `/simulator` — Merge impact & rebase hazard simulator\n\n"
        "Source repository: https://github.com/Aravind0403/vllm_Repo\n"
    )


def _curate_collisions(eng: PRRadarEngine, top_global: int = 1500, per_hotspot: int = 120, top_hotspots: int = 30) -> List[Dict[str, Any]]:
    ordered = sorted(
        eng.collisions,
        key=lambda c: (not c.is_live_race, c.risk_level != "CRITICAL", -c.collision_score),
    )
    selected: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int]] = set()

    def add(c) -> None:
        key = (min(c.pr_a, c.pr_b), max(c.pr_a, c.pr_b))
        if key not in seen:
            seen.add(key)
            selected.append(c.model_dump())

    for c in ordered[:top_global]:
        add(c)

    hotspot_paths = [h.path for h in eng.hotspots[:top_hotspots]]
    counts_by_path: Dict[str, int] = {p: 0 for p in hotspot_paths}
    hotspot_set = set(hotspot_paths)
    for c in ordered:
        matched = [f for f in c.overlapping_files if f in hotspot_set and counts_by_path[f] < per_hotspot]
        if matched:
            add(c)
            for f in matched:
                counts_by_path[f] += 1

    return selected


def export_static_site(eng: PRRadarEngine, out_dir: str | Path) -> Dict[str, Any]:
    """
    Build a self-contained static web bundle (for Hugging Face Static Spaces,
    GitHub Pages, etc.) covering all 6 PR Radar pages and client-side queries.
    """
    import re

    out = Path(out_dir)
    data_dir = out / "static-data"
    data_dir.mkdir(parents=True, exist_ok=True)

    stats = eng.get_summary_stats()
    overview = {
        "repo": eng.repo_slug,
        "total_open_prs": stats["total_open_prs_analyzed"],
        "last_synced_at": stats["last_synced_at"],
        "sync_mode": "static-snapshot",
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

    health = {
        "status": "healthy",
        "repo": eng.repo_slug,
        "prs_indexed": len(eng.prs),
        "last_synced_at": eng.last_synced_at.isoformat() if eng.last_synced_at else None,
        "sync_mode": "static-snapshot",
        "sync_note": "Static snapshot",
        "offline": True,
        "refreshing": False,
        "sync_ttl_minutes": 30.0,
        "version": "1.1.0",
    }

    subsystems = []
    for s in SUBSYSTEM_DEFINITIONS:
        regex = re.compile(s["pattern"])
        count = sum(1 for pr in eng.prs if regex.search(pr.title))
        subsystems.append({**s, "matching_prs_count": count})

    chains = eng.dag.get_dependency_chains() if eng.dag else []
    cycles = eng.dag.get_cycles() if eng.dag else []
    nodes_by_id: Dict[int, Dict[str, Any]] = {}
    for pr in eng.prs:
        nodes_by_id[pr.number] = _serialize_dag_node(eng, pr.number)

    edges = []
    seen_edges: Set[Tuple[int, int]] = set()
    for chain in chains:
        for i in range(len(chain) - 1):
            base_num, waiting_num = chain[i], chain[i + 1]
            for num, is_base in ((base_num, True), (waiting_num, False)):
                if num not in nodes_by_id:
                    nodes_by_id[num] = _serialize_dag_node(eng, num, is_base=is_base)
            edge_key = (waiting_num, base_num)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({"from": waiting_num, "to": base_num, "kind": "stacked_on"})

    dag_payload = {
        "repo": eng.repo_slug,
        "scope": "all",
        "nodes": list(nodes_by_id.values()),
        "edges": edges,
        "chains": chains,
        "cycles": cycles,
    }

    hotspots_payload = []
    for h in eng.hotspots[:200]:
        row = h.model_dump()
        row["recent_pr_numbers"] = sorted(
            h.pr_numbers,
            key=lambda n: eng.analyses[n].staleness_days if n in eng.analyses else 9999.0,
        )
        hotspots_payload.append(row)

    collisions_payload = _curate_collisions(eng)
    reviewers_payload = [rl.model_dump() for rl in eng.reviewer_load[:250]]
    dossiers_payload = {str(pr.number): eng.get_pr_dossier(pr.number) for pr in eng.prs}

    artifacts = {
        "health.json": health,
        "overview.json": overview,
        "subsystems.json": subsystems,
        "dag.json": dag_payload,
        "hotspots.json": hotspots_payload,
        "collisions.json": collisions_payload,
        "reviewers.json": reviewers_payload,
        "dossiers.json": dossiers_payload,
    }
    for filename, payload in artifacts.items():
        with open(data_dir / filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), default=str)

    web_index = Path(__file__).resolve().parent / "web" / "index.html"
    html = web_index.read_text(encoding="utf-8")
    static_flag = "<script>window.PR_RADAR_STATIC = true;</script>\n</head>"
    if "</head>" in html:
        html = html.replace("</head>", static_flag, 1)
    else:
        html = static_flag + html

    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "404.html").write_text(html, encoding="utf-8")
    for route in SPACE_ROUTES:
        route_dir = out / route
        route_dir.mkdir(parents=True, exist_ok=True)
        (route_dir / "index.html").write_text(html, encoding="utf-8")

    (out / "README.md").write_text(
        _space_readme(eng.repo_slug, len(eng.prs), len(chains)),
        encoding="utf-8",
    )

    vercel_config = {
        "cleanUrls": True,
        "trailingSlash": False,
        "rewrites": [
            {
                "source": f"/({'|'.join(SPACE_ROUTES)})",
                "destination": "/index.html",
            }
        ],
    }
    (out / "vercel.json").write_text(
        json.dumps(vercel_config, indent=2) + "\n",
        encoding="utf-8",
    )

    total_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return {
        "out_dir": str(out),
        "prs": len(dossiers_payload),
        "collisions_exported": len(collisions_payload),
        "hotspots_exported": len(hotspots_payload),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }

