# PR Radar — Live PR Engine for vLLM

> **PR dependency DAGs, file-collision radar, change-risk scoring, review-flow tracking, and a merge impact simulator for high-velocity open-source repos.**

`pr-radar` is a maintainer-side intelligence layer built for **vLLM**. From GitHub PR metadata alone it surfaces multi-PR stacks, warns about concurrent file collisions, scores how risky each change is, shows whose move it is in the review loop, and simulates what landing any PR unblocks.

## Who it is for

| User | Question they open it with | Where they look |
|---|---|---|
| **Release manager / core maintainer** | "What do I merge first, and what does that unblock?" | Dependency DAG, Merge Simulator |
| **Area owner** (scheduler, kernels, ROCm...) | "Which open PRs in my area are risky, and which two are about to collide?" | Triage board risk badges, Collision Radar |
| **Reviewer** | "Who is waiting on me, and for how long?" | Reviewer Load, Waiting on Author column |
| **Contributor** | "Is someone already touching the files I am about to change?" | Hotspots, `inspect <pr>` collisions |

One-line pitch: **vLLM has thousands of open PRs; this tells a maintainer which one to merge next, what it unblocks, what it collides with, and whose move it is.** Every PR number links straight to GitHub so the action (merge, re-request review, ping the author) happens there.

---

## Architecture

```
                GitHub GraphQL API
            (full + incremental polling)
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  GitHubFetcher  — JSON cache per repo w/ TTL  │
   │  DependencyParser — stacked-on / depends-on   │
   │  DAGEngine (NetworkX) — stacks, unblockers     │
   │  CollisionDetector — Jaccard + churn hotspots │
   │  StatusTracker — triage status per PR         │
   │  RiskScorer — sensitive paths × churn × cols. │
   │  ReviewFlow — waiting on reviewer/author/merge│
   └───────────────────────┬──────────────────────┘
                           │ PRRadarEngine
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       Rich CLI       FastAPI REST     Web app
                      /api/v1/*        pr_radar/web/
```

There are no webhooks and no database. Ingestion is GraphQL polling with a JSON cache per repo:

| Situation | Behaviour |
|---|---|
| Cache younger than the TTL (default 30 min) | served as-is, no network |
| Cache older than the TTL | **incremental**: fetch only PRs updated since the last sync, merge, and evict PRs that were merged/closed |
| No cache, or `--refresh` | **full**: paginate the whole open set |
| `PR_RADAR_OFFLINE=1` | never touch the network (used by the test suite) |

If a refresh fails the cached data is served and the failure is reported in `sync_info.note`.

---

## Quickstart

```bash
pip install -e ".[dev]"     # or: pip install pydantic networkx requests rich fastapi uvicorn

# First sync (needs a token: GITHUB_TOKEN, GH_TOKEN, or `gh auth login`)
python3 -m pr_radar.cli sync --repo vllm-project/vllm --limit 0     # 0 = every open PR

# Terminal views (analyze the cache as-is)
python3 -m pr_radar.cli triage
python3 -m pr_radar.cli reviewers
python3 -m pr_radar.cli stacks
python3 -m pr_radar.cli collisions
python3 -m pr_radar.cli inspect 61952

# Full-stack server + web app (refreshes incrementally every TTL)
python3 -m pr_radar.cli serve --port 8000
```

Environment variables for the server: `PR_RADAR_REPO` (default `vllm-project/vllm`), `PR_RADAR_SYNC_TTL_MINUTES` (default 30), `PR_RADAR_SYNC_LIMIT` (default 0 = all), `PR_RADAR_OFFLINE`.

---

## REST API

Every endpoint accepts an optional `?repo=owner/name` to switch repositories.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/health` | `GET` | Repo, PRs indexed, last sync time and mode. |
| `/api/v1/overview` | `GET` | Status, risk, and waiting-on breakdowns; top risk PRs; reviewer load; hotspots. |
| `/api/v1/subsystems` | `GET` | vLLM subsystems with active PR counts. |
| `/api/v1/dag` | `GET` | Graph nodes and edges (`?scope=stacks|all`, `?subsystem=`). |
| `/api/v1/collisions` | `GET` | Pairwise PR collisions with live-race classification. |
| `/api/v1/hotspots` | `GET` | Files touched by multiple open PRs. |
| `/api/v1/triage` | `GET` | Kanban columns: `ready_to_merge`, `blocked_by_dep`, `waiting_review`, `waiting_author`, `failing_ci`, `draft`. |
| `/api/v1/risk` | `GET` | PRs ranked by change-risk score (`?min_level=HIGH`). |
| `/api/v1/reviewers` | `GET` | Per-reviewer pending requests, reviews submitted, oldest wait. |
| `/api/v1/prs/{number}` | `GET` | Dossier: blockers, dependents, collisions, risk reasons, review flow. |
| `/api/v1/simulate/merge/{number}` | `GET` | Merge impact: cascade unblocking and conflict hazards. |
| `/api/v1/sync` | `POST` | Incremental refresh (sync). `?force_refresh=true` runs a full re-fetch in the background. |

---

## Signals

### Dependency DAG
Stacks are discovered from PR bodies ("stacked on #123", "depends on", "blocked by", PR URLs) and from branch bases that point at another open PR's head. Foundational PRs are those with dependents and no blockers.

### Collision radar
Pairwise Jaccard file overlap plus shared line churn, bucketed CRITICAL/HIGH/MEDIUM/LOW. A **live race** is a collision where both PRs were updated within 5 days.

### Change-risk score (0-100)
`100 × (0.5 × sensitivity + 0.2 × churn + 0.3 × collision)` where sensitivity comes from a weighted path table (vLLM: scheduler and KV cache 1.0, `csrc/` 0.9, attention 0.8, quantization/platforms/distributed 0.7, engine 0.6, docs 0). Other repos use a generic keyword table. Every score ships with plain-text reasons; nothing is a black box.

### Review flow
From review requests, submitted reviews, last commit, and last comment the engine decides whose move it is:
`REVIEWER` (no review yet, or author pushed after the last review), `AUTHOR` (changes requested or reviewer commented after the author's last activity), `MERGE` (approved and green), `NONE` (draft), plus how many days it has been waiting for that move. Reviewer load aggregates pending requests per reviewer.

Caches written before review data was collected still load; review-flow fields fall back to `updated_at` and are flagged `has_review_signals: false` until the next sync.

---

## Tests

```bash
python3 -m pytest tests/ -v
```

Tests run fully offline. The server tests analyze the committed `data/` cache and skip if it is absent.
