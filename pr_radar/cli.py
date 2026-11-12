from __future__ import annotations

import argparse
import json
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import box

from .engine import PRRadarEngine
from .models import TriageStatus

console = Console()


def get_status_style(status_val: str) -> str:
    styles = {
        TriageStatus.READY_TO_MERGE.value: "bold green",
        TriageStatus.BLOCKED_BY_DEPENDENCY.value: "bold magenta",
        TriageStatus.BLOCKED_BY_CONFLICT.value: "bold red",
        TriageStatus.BLOCKED_BY_CI.value: "red",
        TriageStatus.WAITING_FOR_AUTHOR.value: "yellow",
        TriageStatus.WAITING_FOR_REVIEW.value: "cyan",
        TriageStatus.DRAFT.value: "dim white",
        TriageStatus.STALE.value: "dim yellow",
    }
    return styles.get(status_val, "white")


def cmd_sync(args):
    repo = args.repo
    limit = args.limit
    force = args.refresh

    mode_hint = "full re-fetch" if force else f"incremental if cache older than {args.max_age:g} min"
    with console.status(f"[bold cyan]Syncing PRs for {repo} ({mode_hint}, limit={limit or 'all'})..."):
        engine = PRRadarEngine(repo_slug=repo)
        engine.sync(limit=limit, use_cache=not force, max_age_minutes=args.max_age)

    stats = engine.get_summary_stats()
    info = engine.sync_info
    console.print(
        f"\n[bold green]✓[/bold green] {info.get('mode', 'cache')} sync: "
        f"{stats['total_open_prs_analyzed']} open PRs indexed for [bold cyan]{repo}[/bold cyan] "
        f"({info.get('fetched', 0)} fetched, {info.get('evicted', 0)} closed/merged removed)"
    )
    if info.get("note"):
        console.print(f"[yellow]note:[/yellow] {info['note']}")
    console.print(f"[dim]last synced: {stats['last_synced_at']}[/dim]")
    
    # Print high level summary
    table = Table(title="Repo PR Radar Overview", box=box.ROUNDED)
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="bold yellow")

    for st, count in stats["status_breakdown"].items():
        table.add_row(f"Status: {st}", str(count))
    table.add_row("Foundational PRs (Unblockers)", str(len(stats["foundational_unblockers"])))
    table.add_row("PR Dependency Stacks", str(stats["dependency_stacks_count"]))
    table.add_row("Critical File Collisions", str(stats["critical_file_collisions"]))
    
    console.print(table)


def _load_engine(repo: str) -> PRRadarEngine:
    """Read commands analyze the cache as-is; run `sync` to refresh."""
    engine = PRRadarEngine(repo_slug=repo).sync(limit=0, use_cache=True, max_age_minutes=None)
    if not engine.prs:
        console.print(f"[yellow]No cached PRs for {repo}. Run: python3 -m pr_radar.cli sync --repo {repo}[/yellow]")
    return engine


def _risk_style(level: str) -> str:
    return {"CRITICAL": "bold red", "HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}.get(level, "white")


def _waiting_str(flow) -> str:
    if flow is None:
        return "-"
    colors = {"REVIEWER": "cyan", "AUTHOR": "yellow", "MERGE": "green", "NONE": "dim"}
    c = colors.get(flow.waiting_on, "white")
    label = flow.waiting_on if console.width >= 120 else flow.waiting_on[:3]
    return f"[{c}]{label}[/{c}] {flow.waiting_days:.0f}d"


def cmd_triage(args):
    repo = args.repo
    engine = _load_engine(repo)

    table = Table(
        title=f"Open PR Triage Radar — {repo}",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    narrow = console.width < 120
    table.expand = True
    table.add_column("PR #", justify="right", style="cyan", min_width=5, no_wrap=True)
    if not narrow:
        table.add_column("Author", style="dim", min_width=8, max_width=14, no_wrap=True)
    table.add_column("Status", min_width=9, max_width=16, no_wrap=True)
    table.add_column("CI", justify="center", min_width=1, no_wrap=True)
    if not narrow:
        table.add_column("Review", justify="center", min_width=8, no_wrap=True)
        table.add_column("Stale", justify="right", min_width=5, no_wrap=True)
    table.add_column("Risk", justify="center", min_width=7, no_wrap=True)
    table.add_column("Wait", min_width=7, max_width=11, no_wrap=True)
    table.add_column("Title / Diagnosis", ratio=1, min_width=30, overflow="fold")

    # Sort: Ready first, then Blocked by Dep, Waiting, CI fail, Conflicting, Stale, Draft
    sort_order = [
        TriageStatus.READY_TO_MERGE.value,
        TriageStatus.BLOCKED_BY_DEPENDENCY.value,
        TriageStatus.WAITING_FOR_REVIEW.value,
        TriageStatus.WAITING_FOR_AUTHOR.value,
        TriageStatus.BLOCKED_BY_CI.value,
        TriageStatus.BLOCKED_BY_CONFLICT.value,
        TriageStatus.STALE.value,
        TriageStatus.DRAFT.value,
    ]

    analyses = sorted(
        engine.analyses.values(),
        key=lambda a: (
            sort_order.index(a.triage_status.value)
            if a.triage_status.value in sort_order
            else 99,
            a.staleness_days,
        ),
    )

    for item in analyses:
        st_val = item.triage_status.value
        style = get_status_style(st_val)
        
        ci_str = "[green]✓[/green]" if item.pr.ci_status == "SUCCESS" else (
            "[red]✗[/red]" if item.pr.ci_status == "FAILURE" else "[yellow]⋯[/yellow]"
        )
        
        rev_str = item.pr.review_decision or "-"
        if rev_str == "APPROVED":
            rev_str = "[green]APPROVED[/green]"
        elif rev_str == "CHANGES_REQUESTED":
            rev_str = "[red]CHANGES[/red]"
        elif rev_str == "REVIEW_REQUIRED":
            rev_str = "[yellow]REQUIRED[/yellow]"

        title_len = 42 if narrow else 60
        title = item.pr.title if len(item.pr.title) <= title_len else item.pr.title[:title_len] + "..."
        by = f" [dim]by {item.pr.author[:12]}[/dim]" if narrow else ""
        diag = f"[bold]{title}[/bold]{by}\n[dim]↳ {item.primary_blocker_reason}[/dim]"
        if item.collision_prs:
            diag += f"\n[red]⚠ Collides with: {', '.join(f'#{p}' for p in item.collision_prs[:3])}[/red]"

        risk = item.risk
        risk_str = f"[{_risk_style(risk.level)}]{risk.score} {risk.level[:4 if not narrow else 1]}[/{_risk_style(risk.level)}]" if risk else "-"

        short_status = st_val.replace("BLOCKED_BY_", "BLK:").replace("WAITING_", "WAIT:").replace("READY_TO_MERGE", "READY")
        row = [str(item.pr.number)]
        if not narrow:
            row.append(item.pr.author[:13])
        row += [f"[{style}]{short_status if narrow else st_val}[/{style}]", ci_str]
        if not narrow:
            row += [rev_str, f"{item.staleness_days:.0f}d"]
        row += [risk_str, _waiting_str(item.review_flow), diag]
        table.add_row(*row)

    console.print(table)


def cmd_stacks(args):
    repo = args.repo
    engine = _load_engine(repo)
    chains = engine.dag.get_dependency_chains() if engine.dag else []

    if not chains:
        console.print(f"[yellow]No explicit PR dependency stacks discovered in the cached PRs for {repo}.[/yellow]")
        return

    console.print(f"\n[bold magenta]=== Discovered PR Stacks & Merge Trains ({len(chains)} chains) ===[/bold magenta]\n")
    for i, chain in enumerate(chains, 1):
        tree = Tree(f"[bold cyan]Stack #{i}[/bold cyan] ({len(chain)} PRs in sequence)")
        curr_node = tree
        for j, pr_num in enumerate(chain):
            pr = engine.pr_map.get(pr_num)
            analysis = engine.analyses.get(pr_num)
            node_data = engine.dag.graph.nodes.get(pr_num, {})
            
            if pr:
                status_str = f"[{get_status_style(analysis.triage_status.value)}]{analysis.triage_status.value}[/]"
                ci_str = "CI green" if pr.ci_status == "SUCCESS" else f"CI: {pr.ci_status}"
                label = f"[bold white]PR #{pr_num}[/bold white]: {pr.title[:50]}... ({status_str}, {ci_str})"
            else:
                title = node_data.get("title", f"PR #{pr_num}")
                state = node_data.get("state", "UNKNOWN")
                state_style = "bold green" if state == "MERGED" else ("bold yellow" if state == "OPEN" else "dim")
                label = f"[bold white]PR #{pr_num}[/bold white]: {title[:50]}... ([{state_style}]{state}[/{state_style}])"
            
            if j == 0:
                curr_node = tree.add(f"[bold green]BASE (Unblocker):[/bold green] {label}")
            else:
                curr_node = curr_node.add(f"[bold yellow]WAITING:[/bold yellow] {label}")
        
        console.print(tree)
        console.print()


def cmd_collisions(args):
    repo = args.repo
    engine = _load_engine(repo)

    # 1. Hotspot files
    hotspot_table = Table(
        title=f"🔥 Top Conflict Hotspot Files — {repo}",
        box=box.ROUNDED,
        header_style="bold red",
    )
    hotspot_table.add_column("File Path", style="cyan")
    hotspot_table.add_column("Open PRs Touching", justify="center", style="bold yellow")
    hotspot_table.add_column("PR Numbers", style="dim")
    hotspot_table.add_column("Total Churn (Lines)", justify="right", style="green")

    for h in engine.hotspots[:10]:
        pr_str = ", ".join(f"#{num}" for num in h.pr_numbers[:6])
        if len(h.pr_numbers) > 6:
            pr_str += f" (+{len(h.pr_numbers) - 6} more)"
        hotspot_table.add_row(h.path, str(h.open_pr_count), pr_str, str(h.total_churn))

    console.print(hotspot_table)
    console.print()

    # 2. Pairwise collisions
    col_table = Table(
        title=f"⚠️ Pairwise PR Collision Matrix — {repo}",
        box=box.ROUNDED,
        header_style="bold yellow",
    )
    col_table.add_column("PR A", style="cyan", width=14)
    col_table.add_column("PR B", style="cyan", width=14)
    col_table.add_column("Risk Level", width=12)
    col_table.add_column("Activity", width=16)
    col_table.add_column("Score", justify="right", width=6)
    col_table.add_column("Shared Churn", justify="right", width=12)
    col_table.add_column("Overlapping Files", style="dim", width=45)

    for c in engine.collisions[:15]:
        risk_style = "bold red" if c.risk_level == "CRITICAL" else ("bold yellow" if c.risk_level == "HIGH" else "white")
        files_str = ", ".join(f.split("/")[-1] for f in c.overlapping_files[:4])
        if len(c.overlapping_files) > 4:
            files_str += f" (+{len(c.overlapping_files) - 4} more)"

        activity_str = "[bold red]🔥 LIVE RACE[/bold red]" if c.is_live_race else (
            f"[dim]Stale ({max(c.pr_a_idle_days, c.pr_b_idle_days):.0f}d)[/dim]"
        )

        col_table.add_row(
            f"#{c.pr_a} ({c.pr_a_idle_days:.0f}d)",
            f"#{c.pr_b} ({c.pr_b_idle_days:.0f}d)",
            f"[{risk_style}]{c.risk_level}[/{risk_style}]",
            activity_str,
            f"{c.collision_score:.2f}",
            f"{c.shared_churn} lines",
            files_str,
        )

    console.print(col_table)


def cmd_inspect(args):
    repo = args.repo
    pr_num = args.pr_number
    engine = _load_engine(repo)

    dossier = engine.get_pr_dossier(pr_num)
    if not dossier:
        console.print(f"[bold red]PR #{pr_num} not found in synced PRs for {repo}. Try running with --refresh.[/bold red]")
        return

    st_style = get_status_style(dossier["status"])
    content = f"""
[bold cyan]Title:[/bold cyan] {dossier['title']}
[bold cyan]Author:[/bold cyan] {dossier['author']}
[bold cyan]URL:[/bold cyan] {dossier['url']}
[bold cyan]Triage Status:[/bold cyan] [{st_style}]{dossier['status']}[/{st_style}]
[bold cyan]Diagnosis:[/bold cyan] {dossier['reason']}

[bold magenta]Technical Signals:[/bold magenta]
• CI Status: {dossier['ci_status']}
• Review Decision: {dossier['review_decision']}
• Git Mergeable: {dossier['mergeable']}
• Draft: {dossier['is_draft']}
• Staleness: {dossier['staleness_days']} days since last activity
• Files Changed: {dossier['files_changed_count']} files

[bold yellow]Dependency Graph:[/bold yellow]
• Blocked by (upstream): {dossier['blocked_by'] or 'None (Unblocked)'}
• Blocks (downstream): {dossier['blocks'] or 'None'}
"""
    rs = _risk_style(dossier["risk_level"])
    content += f"""
[bold red]Risk Assessment:[/bold red] [{rs}]{dossier['risk_score']}/100 {dossier['risk_level']}[/{rs}]
"""
    for reason in dossier["risk_reasons"] or ["No risk signals"]:
        content += f"• {reason}\n"

    flow = dossier.get("review_flow") or {}
    if flow:
        content += f"""
[bold cyan]Review Flow:[/bold cyan] waiting on [bold]{flow['waiting_on']}[/bold] for {flow['waiting_days']:.0f} days
• Reviews: {flow['review_count']} (by {', '.join(flow['reviewers_engaged']) or 'nobody yet'})
• Requested reviewers: {', '.join(flow['requested_reviewers']) or 'none'}
• Last review: {flow['last_review_state'] or '-'} by {flow['last_reviewer'] or '-'}
"""
        if not flow.get("has_review_signals"):
            content += "[dim]• (no review data in cache; run `sync` to fetch review events)[/dim]\n"

    console.print(Panel(content, title=f"Dossier: PR #{pr_num}", border_style="cyan"))

    if dossier["collisions"]:
        console.print("[bold red]Colliding PRs touching the same files:[/bold red]")
        for c in dossier["collisions"]:
            console.print(f"  • [bold]PR #{c['colliding_pr']}[/bold] ([red]{c['risk_level']}[/red], score {c['collision_score']}): {', '.join(c['overlapping_files'])}")


def cmd_reviewers(args):
    repo = args.repo
    engine = _load_engine(repo)

    if not engine.reviewer_load:
        console.print("[yellow]No review data in cache. Run `sync` to fetch review requests and reviews.[/yellow]")
        return

    table = Table(title=f"Reviewer Load — {repo}", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Reviewer", style="bold white", width=22)
    table.add_column("Pending", justify="right", style="bold yellow", width=8)
    table.add_column("Oldest wait", justify="right", width=11)
    table.add_column("Reviewed", justify="right", width=9)
    table.add_column("Approved", justify="right", style="green", width=9)
    table.add_column("Changes req.", justify="right", style="red", width=12)
    table.add_column("Pending PRs", style="dim", width=40)

    for rl in engine.reviewer_load[: args.limit]:
        prs_str = ", ".join(f"#{n}" for n in rl.pending_pr_numbers[:6])
        if len(rl.pending_pr_numbers) > 6:
            prs_str += f" (+{len(rl.pending_pr_numbers) - 6})"
        table.add_row(
            rl.reviewer, str(rl.pending_requests), f"{rl.oldest_pending_days:.0f}d",
            str(rl.reviews_submitted), str(rl.approvals), str(rl.changes_requested), prs_str,
        )
    console.print(table)

    stats = engine.get_summary_stats()
    w = stats["waiting_on_breakdown"]
    console.print(
        f"\n[dim]Waiting on: reviewer {w.get('REVIEWER', 0)} · author {w.get('AUTHOR', 0)} · "
        f"merge {w.get('MERGE', 0)} · draft {w.get('NONE', 0)} · "
        f"median reviewer wait {stats['median_reviewer_wait_days']}d[/dim]"
    )


def cmd_export(args):
    repo = args.repo
    out_file = args.out
    engine = PRRadarEngine(repo_slug=repo).sync(use_cache=True)

    export_data = {
        "summary": engine.get_summary_stats(),
        "prs": [engine.get_pr_dossier(pr.number) for pr in engine.prs],
        "hotspots": [h.model_dump() for h in engine.hotspots],
        "collisions": [c.model_dump() for c in engine.collisions],
        "chains": engine.dag.get_dependency_chains() if engine.dag else [],
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    console.print(f"[bold green]✓[/bold green] Exported comprehensive PR Radar data to [bold cyan]{out_file}[/bold cyan]")


def cmd_dashboard(args):
    import webbrowser
    import http.server
    import socketserver
    import threading
    from pathlib import Path

    repo = args.repo
    engine = PRRadarEngine(repo_slug=repo).sync(limit=0, use_cache=not args.refresh, max_age_minutes=None)

    export_data = {
        "summary": engine.get_summary_stats(),
        "prs": [engine.get_pr_dossier(pr.number) for pr in engine.prs],
        "hotspots": [h.model_dump() for h in engine.hotspots],
        "collisions": [c.model_dump() for c in engine.collisions],
        "chains": engine.dag.get_dependency_chains() if engine.dag else [],
    }

    base_dir = Path(__file__).resolve().parent.parent
    data_json_path = base_dir / "pr_radar_data.json"
    html_path = base_dir / "dashboard.html"

    # Write data JSON
    with open(data_json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    # Also embed JSON into dashboard.html so file:// protocol works with zero CORS restrictions
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        json_str = json.dumps(export_data)
        replacement = f"window.__EMBEDDED_PR_RADAR_DATA__ = {json_str};\n"
        if "window.__EMBEDDED_PR_RADAR_DATA__ =" in content:
            import re
            content = re.sub(
                r"window\.__EMBEDDED_PR_RADAR_DATA__\s*=\s*.*?;",
                f"window.__EMBEDDED_PR_RADAR_DATA__ = {json_str};",
                content,
                flags=re.DOTALL,
            )
        else:
            content = content.replace(
                "let DATA = null;",
                f"let DATA = null;\n    window.__EMBEDDED_PR_RADAR_DATA__ = {json_str};"
            )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)

    console.print(f"[bold green]✓[/bold green] Prepared interactive dashboard for [bold cyan]{repo}[/bold cyan]")
    
    # Open dashboard in browser
    url = f"file://{html_path.resolve()}"
    console.print(f"[bold cyan]Opening dashboard in browser:[/bold cyan] {url}")
    webbrowser.open(url)


def cmd_serve(args):
    import uvicorn
    import webbrowser
    import threading
    import time

    host = args.host
    port = args.port

    url = f"http://{host}:{port}"
    console.print(f"\n[bold green]🚀 Launching PR Radar Full-Stack Server on[/bold green] [bold cyan]{url}[/bold cyan]")
    console.print(f"[dim]• Interactive REST API Docs: {url}/docs[/dim]")
    console.print(f"[dim]• Live Web Application:      {url}/[/dim]\n")

    if not args.no_browser:
        def open_browser():
            time.sleep(1.0)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("pr_radar.server:app", host=host, port=port, log_level="info")


def main():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--repo",
        default="vllm-project/vllm",
        help="Target repository (e.g. vllm-project/vllm, sgl-project/sglang)",
    )

    parser = argparse.ArgumentParser(description="PR Live Merges Radar & Dependency Engine", parents=[base_parser])
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    serve_p = subparsers.add_parser("serve", help="Launch the full-stack FastAPI server & web application", parents=[base_parser])
    serve_p.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    serve_p.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_p.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")

    # sync
    sync_p = subparsers.add_parser("sync", help="Sync PR data from GitHub", parents=[base_parser])
    sync_p.add_argument("--limit", type=int, default=500, help="Max open PRs to index (0 = all)")
    sync_p.add_argument("--refresh", action="store_true", help="Discard cache and re-fetch everything")
    sync_p.add_argument("--max-age", type=float, default=30.0, help="Cache age in minutes before an incremental refresh")

    # triage
    subparsers.add_parser("triage", help="View triage table of open PRs", parents=[base_parser])

    # stacks
    subparsers.add_parser("stacks", help="View PR dependency stacks and merge chains", parents=[base_parser])

    # collisions
    subparsers.add_parser("collisions", help="View file collision radar and hotspot files", parents=[base_parser])

    # reviewers
    rev_p = subparsers.add_parser("reviewers", help="Reviewer workload: pending requests and review activity", parents=[base_parser])
    rev_p.add_argument("--limit", type=int, default=20, help="Rows to show")

    # inspect
    inspect_p = subparsers.add_parser("inspect", help="Inspect a specific PR dossier", parents=[base_parser])
    inspect_p.add_argument("pr_number", type=int, help="PR number to inspect")

    # export
    export_p = subparsers.add_parser("export", help="Export full data engine state to JSON", parents=[base_parser])
    export_p.add_argument("--out", default="pr_radar_export.json", help="Output file path")

    # dashboard
    dash_p = subparsers.add_parser("dashboard", help="Launch the interactive visual web dashboard", parents=[base_parser])
    dash_p.add_argument("--refresh", action="store_true", help="Bypass local cache and re-fetch from GitHub")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "triage":
        cmd_triage(args)
    elif args.command == "stacks":
        cmd_stacks(args)
    elif args.command == "collisions":
        cmd_collisions(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "reviewers":
        cmd_reviewers(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
