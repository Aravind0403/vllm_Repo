from __future__ import annotations

from typing import List, Dict, Set, Tuple
from collections import defaultdict
from .models import PullRequest, FileCollision, HotspotFile


class CollisionDetector:
    """
    Analyzes touched files across all open PRs to calculate:
    1. Pairwise PR collision risk (which PRs will likely conflict with each other)
    2. Hotspot files (files undergoing heavy concurrent modification)
    """

    @staticmethod
    def calculate_collisions(prs: List[PullRequest]) -> List[FileCollision]:
        """
        Calculates pairwise file collisions across all provided PRs.
        """
        collisions: List[FileCollision] = []
        n = len(prs)

        # Precompute file maps: pr.number -> {path: file_obj}
        pr_files_map: Dict[int, Dict[str, any]] = {
            pr.number: {f.path: f for f in pr.files} for pr in prs
        }

        for i in range(n):
            pr_a = prs[i]
            files_a = set(pr_files_map[pr_a.number].keys())
            if not files_a:
                continue

            for j in range(i + 1, n):
                pr_b = prs[j]
                files_b = set(pr_files_map[pr_b.number].keys())
                if not files_b:
                    continue

                overlap = files_a.intersection(files_b)
                if not overlap:
                    continue

                union_len = len(files_a.union(files_b))
                jaccard = len(overlap) / union_len if union_len > 0 else 0.0

                shared_churn = 0
                for f_path in overlap:
                    fa = pr_files_map[pr_a.number].get(f_path)
                    fb = pr_files_map[pr_b.number].get(f_path)
                    shared_churn += (fa.changes if fa else 0) + (fb.changes if fb else 0)

                # Collision score: combines Jaccard overlap + churn factor
                churn_factor = min(1.0, shared_churn / 200.0)
                score = round((0.6 * jaccard) + (0.4 * churn_factor), 3)

                # Risk classification
                if len(overlap) >= 3 or (len(overlap) >= 1 and shared_churn >= 150):
                    risk = "CRITICAL"
                elif len(overlap) >= 2 or shared_churn >= 60:
                    risk = "HIGH"
                elif len(overlap) == 1 and shared_churn >= 15:
                    risk = "MEDIUM"
                else:
                    risk = "LOW"

                collisions.append(
                    FileCollision(
                        pr_a=pr_a.number,
                        pr_b=pr_b.number,
                        overlapping_files=sorted(list(overlap)),
                        collision_score=score,
                        risk_level=risk,
                        shared_churn=shared_churn,
                    )
                )

        # Sort collisions from highest risk score downwards
        collisions.sort(key=lambda c: (c.risk_level == "CRITICAL", c.risk_level == "HIGH", c.collision_score), reverse=True)
        return collisions

    @staticmethod
    def identify_hotspots(prs: List[PullRequest], min_prs: int = 2) -> List[HotspotFile]:
        """
        Identifies files that are concurrently modified by multiple open PRs.
        """
        file_to_prs: Dict[str, Set[int]] = defaultdict(set)
        file_to_churn: Dict[str, int] = defaultdict(int)

        for pr in prs:
            for f in pr.files:
                file_to_prs[f.path].add(pr.number)
                file_to_churn[f.path] += f.changes

        hotspots: List[HotspotFile] = []
        for path, pr_nums in file_to_prs.items():
            if len(pr_nums) >= min_prs:
                hotspots.append(
                    HotspotFile(
                        path=path,
                        open_pr_count=len(pr_nums),
                        pr_numbers=sorted(list(pr_nums)),
                        total_churn=file_to_churn[path],
                    )
                )

        # Sort by most PRs touching the file, then by churn
        hotspots.sort(key=lambda h: (h.open_pr_count, h.total_churn), reverse=True)
        return hotspots

