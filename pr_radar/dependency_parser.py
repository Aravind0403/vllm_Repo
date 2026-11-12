from __future__ import annotations

import re
from typing import List, Dict, Optional, Set
from .models import PullRequest, DependencyRelation, DependencyKind


class DependencyParser:
    """
    Parses PR bodies, titles, and branch references to discover
    dependency relationships between PRs.
    """

    # Primary dependency patterns with capturing groups.
    # Require either `#123` or explicit `PR #123` / `PR 123` so plain counts
    # ("requires 2 GPUs", "parent 512 MiB", "parent 670aa7ec") are ignored.
    _REF_LIST = r"((?:pr\s+#?|#)[0-9]+\b(?:\s*(?:,|and|&|/)\s*(?:pr\s+#?|#)[0-9]+\b)*)"

    PATTERNS = [
        # Stacked on #123, #456, and #789 (skip compound words like foundation-model-stack)
        (
            re.compile(
                rf"(?i)(?<![-\w])(?:stacked\s+(?:on|upon)|stack\s*:)\s*{_REF_LIST}",
                re.MULTILINE,
            ),
            DependencyKind.STACKED_ON,
            1.0,
        ),
        # Depends on #123, Depends on PR #123
        (
            re.compile(
                rf"(?i)\b(?:depends\s+(?:on|upon)|dependent\s+on)\s*{_REF_LIST}",
                re.MULTILINE,
            ),
            DependencyKind.DEPENDS_ON,
            1.0,
        ),
        # Blocked by #123, Blocked on #123
        (
            re.compile(
                rf"(?i)\b(?:blocked\s+(?:by|on))\s*{_REF_LIST}",
                re.MULTILINE,
            ),
            DependencyKind.BLOCKED_BY,
            1.0,
        ),
        # Requires #123
        (
            re.compile(
                rf"(?i)\b(?:requires|prerequisite\s*:?)\s*{_REF_LIST}",
                re.MULTILINE,
            ),
            DependencyKind.REQUIRES,
            0.9,
        ),
        # Based on #123
        (
            re.compile(
                rf"(?i)\b(?:based\s+on|parent\s+pr\s*:?|parent\s*:)\s*{_REF_LIST}",
                re.MULTILINE,
            ),
            DependencyKind.STACKED_ON,
            0.85,
        ),
        # Follow-up to #123
        (
            re.compile(
                rf"(?i)\b(?:follow[-\s]?up\s+(?:to|of))\s*{_REF_LIST}",
                re.MULTILINE,
            ),
            DependencyKind.RELATED,
            0.8,
        ),
    ]

    NEGATION_PREFIX = re.compile(
        r"(?i)\b(?:not|no\s+longer|doesn't|does\s+not|without|never)\s+$"
    )

    URL_PATTERN = re.compile(
        r"https?://github\.com/([\w\-\.]+)/([\w\-\.]+)/pull/([0-9]+)"
    )

    NUMBER_EXTRACTOR = re.compile(r"(?:(?<![\w./-])#|(?i:\bpr\s+))([0-9]+)\b")

    @classmethod
    def parse_pr(
        cls,
        pr: PullRequest,
        known_pr_numbers: Optional[Set[int]] = None,
        branch_to_pr: Optional[Dict[str, int]] = None,
    ) -> List[DependencyRelation]:
        """
        Extract all dependencies where `pr` depends on other PRs.
        """
        relations: List[DependencyRelation] = []
        seen_targets: Set[int] = {pr.number}  # Disallow self-dependencies

        # 1. Branch Base check: If base_ref is not main/master, check if it targets another PR's head_ref
        if branch_to_pr and pr.base_ref and pr.base_ref not in ("main", "master", "dev", "develop"):
            target_pr = branch_to_pr.get(pr.base_ref)
            if target_pr and target_pr not in seen_targets:
                relations.append(
                    DependencyRelation(
                        source_pr=pr.number,
                        target_pr=target_pr,
                        kind=DependencyKind.BRANCH_BASE,
                        context_snippet=f"Base branch '{pr.base_ref}' targets PR #{target_pr}",
                        confidence=1.0,
                    )
                )
                seen_targets.add(target_pr)

        body_text = pr.body or ""
        lines = body_text.splitlines()

        # 2. Heuristic text search line by line
        repo_lower = pr.repo.lower() if pr.repo else ""
        for line in lines:
            # Check dependency keywords
            for pattern, kind, conf in cls.PATTERNS:
                matches = pattern.finditer(line)
                for m in matches:
                    if cls.NEGATION_PREFIX.search(line[: m.start()]):
                        continue
                    raw_numbers = m.group(1)
                    nums = cls.NUMBER_EXTRACTOR.findall(raw_numbers)
                    for n_str in nums:
                        try:
                            n = int(n_str)
                        except ValueError:
                            continue
                        if n in seen_targets:
                            continue
                        if known_pr_numbers and n not in known_pr_numbers:
                            # If we have a catalog of known PRs, ignore issue numbers that aren't PRs
                            continue
                        
                        seen_targets.add(n)
                        relations.append(
                            DependencyRelation(
                                source_pr=pr.number,
                                target_pr=n,
                                kind=kind,
                                context_snippet=line.strip()[:140],
                                confidence=conf,
                            )
                        )

            # Check full GitHub PR URLs in lines containing dependency-like phrases
            if any(k in line.lower() for k in ["depend", "block", "stack", "require", "after", "parent"]):
                url_matches = cls.URL_PATTERN.findall(line)
                for owner, repo_name, n_str in url_matches:
                    if repo_lower and f"{owner}/{repo_name}".lower() != repo_lower:
                        continue
                    try:
                        n = int(n_str)
                    except ValueError:
                        continue
                    if n in seen_targets:
                        continue
                    if known_pr_numbers and n not in known_pr_numbers:
                        continue
                    
                    seen_targets.add(n)
                    relations.append(
                        DependencyRelation(
                            source_pr=pr.number,
                            target_pr=n,
                            kind=DependencyKind.DEPENDS_ON,
                            context_snippet=line.strip()[:140],
                            confidence=0.9,
                        )
                    )

        return relations

