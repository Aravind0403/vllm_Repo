from __future__ import annotations

from typing import List, Dict, Set, Optional, Tuple
import networkx as nx
from .models import PullRequest, DependencyRelation


class DAGEngine:
    """
    Manages the directed acyclic graph (DAG) of PR dependencies.
    Convention: Edge (U, V) indicates U DEPENDS ON V (V blocks U).
    Therefore:
      - V is upstream (must land first).
      - U is downstream (waiting on V).
    """

    def __init__(self, prs: List[PullRequest], relations: List[DependencyRelation]):
        self.prs: Dict[int, PullRequest] = {pr.number: pr for pr in prs}
        self.relations = relations
        self.graph = nx.DiGraph()

        # Add all PRs as nodes
        for pr in prs:
            self.graph.add_node(
                pr.number,
                title=pr.title,
                author=pr.author,
                is_draft=pr.is_draft,
                state=pr.state,
                review_decision=pr.review_decision,
                ci_status=pr.ci_status,
                mergeable=pr.mergeable,
            )

        # Add dependency edges
        for rel in relations:
            # Source depends on Target
            if not self.graph.has_node(rel.target_pr):
                # Add ghost node if target PR is known by number but not in current batch
                self.graph.add_node(rel.target_pr, title="External/Historical PR", author="", is_ghost=True)

            self.graph.add_edge(
                rel.source_pr,
                rel.target_pr,
                kind=rel.kind.value,
                context=rel.context_snippet,
                confidence=rel.confidence,
            )

    def get_cycles(self) -> List[List[int]]:
        """Detect any circular dependency cycles."""
        try:
            return list(nx.simple_cycles(self.graph))
        except Exception:
            return []

    def get_direct_blockers(self, pr_number: int) -> List[int]:
        """Returns direct PRs that this PR depends on."""
        if not self.graph.has_node(pr_number):
            return []
        return list(self.graph.successors(pr_number))

    def get_direct_dependents(self, pr_number: int) -> List[int]:
        """Returns direct PRs that depend on this PR."""
        if not self.graph.has_node(pr_number):
            return []
        return list(self.graph.predecessors(pr_number))

    def get_all_blockers(self, pr_number: int) -> Set[int]:
        """Returns all transitive PRs that must land before this PR can land."""
        if not self.graph.has_node(pr_number):
            return set()
        return nx.descendants(self.graph, pr_number)

    def get_all_dependents(self, pr_number: int) -> Set[int]:
        """Returns all transitive PRs waiting for this PR."""
        if not self.graph.has_node(pr_number):
            return set()
        return nx.ancestors(self.graph, pr_number)

    def get_foundational_prs(self) -> List[int]:
        """
        PRs that have dependents (others waiting on them)
        AND have NO blockers themselves.
        Merging these unblocks downstream PRs immediately.
        """
        foundational = []
        for node in self.graph.nodes():
            has_dependents = self.graph.in_degree(node) > 0
            has_no_blockers = self.graph.out_degree(node) == 0
            if has_dependents and has_no_blockers:
                foundational.append(node)
        return foundational

    def get_dependency_chains(self) -> List[List[int]]:
        """
        Extracts maximal dependency chains (stacks / merge trains).
        e.g., [Bottom_PR, Middle_PR, Top_PR]
        """
        # Find all weakly connected components with >= 2 nodes
        subgraphs = [
            self.graph.subgraph(c).copy()
            for c in nx.weakly_connected_components(self.graph)
            if len(c) > 1
        ]

        chains = []
        for sg in subgraphs:
            # Find leaves (nodes with out_degree == 0, i.e. base of the stack)
            bases = [n for n in sg.nodes() if sg.out_degree(n) == 0]
            # Find tips (nodes with in_degree == 0, i.e. top of the stack)
            tips = [n for n in sg.nodes() if sg.in_degree(n) == 0]

            for tip in tips:
                for base in bases:
                    if nx.has_path(sg, tip, base):
                        for path in nx.all_simple_paths(sg, tip, base):
                            # Reverse path so it reads: [Base, ..., Tip]
                            chains.append(list(reversed(path)))

            # Surface cycles (e.g. A <-> B) that have no tips/bases so they aren't hidden
            if not bases or not tips:
                for cyc in nx.simple_cycles(sg):
                    if cyc:
                        chains.append(sorted(cyc))

        # Deduplicate and sort by length
        unique_chains: List[List[int]] = []
        for chain in chains:
            if not any(
                len(chain) < len(other) and all(x in other for x in chain)
                for other in chains
            ):
                if chain not in unique_chains:
                    unique_chains.append(chain)

        unique_chains.sort(key=len, reverse=True)
        return unique_chains

