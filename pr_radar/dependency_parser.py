from __future__ import annotations

import re
from typing import List, Dict, Optional, Set
from .models import PullRequest, DependencyRelation, DependencyKind


class DependencyParser:
    """
    Parses PR bodies, titles, and branch references to discover
    dependency relationships between PRs.
    """

    # Primary dependency patterns with capturing groups
    PATTERNS = [
        # Stacked on #123, #456, and #789
        (
            re.compile(
                r"(?i)\b(?:stacked\s+(?:on|upon)|stack\s*:?)\s*([#0-9\s,and&]+)",
                re.MULTILINE,
            ),
            DependencyKind.STACKED_ON,
            1.0,
        ),
        # Depends on #123, Depends on PR #123
        (
            re.compile(
                r"(?i)\b(?:depends\s+(?:on|upon)|dependent\s+on)\s*(?:pr\s*)?([#0-9\s,and&]+)",
                re.MULTILINE,
            ),
            DependencyKind.DEPENDS_ON,
            1.0,
        ),
        # Blocked by #123, Blocked on #123
        (
            re.compile(
                r"(?i)\b(?:blocked\s+(?:by|on))\s*(?:pr\s*)?([#0-9\s,and&]+)",
                re.MULTILINE,
            ),
            DependencyKind.BLOCKED_BY,
            1.0,
        ),
        # Requires #123
        (
            re.compile(
                r"(?i)\b(?:requires|prerequisite\s*:?)\s*(?:pr\s*)?([#0-9\s,and&]+)",
                re.MULTILINE,
            ),
            DependencyKind.REQUIRES,
            0.9,
        ),
        # Based on #123
        (
            re.compile(
                r"(?i)\b(?:based\s+on|parent\s+pr\s*:?|parent\s*:?)\s*(?:pr\s*)?([#0-9\s,and&]+)",
                re.MULTILINE,
            ),
            DependencyKind.STACKED_ON,
            0.85,
        ),
        # Follow-up to #123
        (
            re.compile(
                r"(?i)\b(?:follow[-\s]?up\s+(?:to|of))\s*(?:pr\s*)?([#0-9\s,and&]+)",
                re.MULTILINE,
            ),
            DependencyKind.RELATED,
            0.8,
        ),
    ]

    URL_PATTERN = re.compile(
        r"https?://github\.com/(?:[\w\-\.]+)/(?:[\w\-\.]+)/pull/([0-9]+)"
    )

    NUMBER_EXTRACTOR = re.compile(r"#?([0-9]+)")

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
        for line in lines:
            # Check dependency keywords
            for pattern, kind, conf in cls.PATTERNS:
                matches = pattern.finditer(line)
                for m in matches:
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
                for n_str in url_matches:
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

