"""
knowledge_graph.py  –  Citation / similarity graph between papers
-----------------------------------------------------------------
Builds an undirected weighted graph linking papers by embedding similarity.
Used in both app.py (Streamlit) and main.py (CLI).
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class CitationGraph:

    def __init__(self):
        self.graph = nx.Graph()

    # ── basic operations ──────────────────────────────────────────────────────

    def add_paper(self, paper_name: str) -> None:
        """Add a paper as a node."""
        self.graph.add_node(paper_name)

    def add_relationship(self, paper1: str, paper2: str, weight: float = 1.0) -> None:
        """Add a weighted edge between two papers."""
        self.graph.add_edge(paper1, paper2, weight=round(weight, 4))

    def show_connections(self) -> list:
        """Return all edges as (paper1, paper2) tuples."""
        return list(self.graph.edges())

    # ── similarity linking ────────────────────────────────────────────────────

    def find_similar_papers(
        self,
        paper_names: list,
        paper_embeddings: list,
        threshold: float = 0.70,
    ) -> None:
        """
        Auto-connect papers whose mean embedding cosine similarity >= threshold.

        Parameters
        ----------
        paper_names      : list of paper display names (same order as embeddings)
        paper_embeddings : list of 1-D numpy arrays (one mean embedding per paper)
        threshold        : min cosine similarity to draw an edge (0.0 – 1.0)
        """
        if len(paper_names) != len(paper_embeddings):
            raise ValueError(
                f"paper_names ({len(paper_names)}) and "
                f"paper_embeddings ({len(paper_embeddings)}) must have same length."
            )

        for name in paper_names:
            self.add_paper(name)

        matrix     = np.vstack(paper_embeddings)   # (n_papers, dim)
        sim_matrix = cosine_similarity(matrix)      # (n_papers, n_papers)

        n = len(paper_names)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score >= threshold:
                    self.add_relationship(paper_names[i], paper_names[j], weight=score)

    # ── analytics ─────────────────────────────────────────────────────────────

    def get_paper_stats(self) -> dict:
        """Return degree-centrality score for each paper (higher = more connected)."""
        if len(self.graph.nodes) == 0:
            return {}
        return nx.degree_centrality(self.graph)

    def most_connected_paper(self):
        """Return the name of the most central paper, or None if graph is empty."""
        stats = self.get_paper_stats()
        if not stats:
            return None
        return max(stats, key=stats.get)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict with nodes and weighted edges."""
        return {
            "nodes": list(self.graph.nodes()),
            "edges": [
                {"source": u, "target": v, "weight": d.get("weight", 1.0)}
                for u, v, d in self.graph.edges(data=True)
            ],
        }