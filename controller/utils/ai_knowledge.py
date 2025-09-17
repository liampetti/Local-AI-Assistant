"""
Handler for centralized AI knowledge graph.

This module provides all knowledge graph functionality 
"""
import re
import math
import json
from typing import List, Dict, Any, Tuple, Set
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph
from ollama import Client

from config import config
from utils.system_prompts import getPlannerSystemPrompt


class KnowledgeHandler:
    """Centralized knowledge handler"""
    
    def __init__(self):
        self.logger = config.get_logger("KGLLM")
        self.sys_planner = getPlannerSystemPrompt()

    # -----------------------------
    # Knowledge Graph Utilities
    # -----------------------------

    def build_initial_graph(self) -> nx.DiGraph:
        """
        Create a small family knowledge graph with imagined facts.
        Nodes and edges carry attributes; edges have a 'relation' and 'weight'.
        """
        G = nx.DiGraph()
        # People
        people = ["Alice Johnson", "Mark Johnson", "Bob Johnson", "Ella Johnson", "Grandma Rose"]
        for p in people:
            G.add_node(p, type="Person", weight=1.0)

        # Places / Orgs
        G.add_node("Maple Street 42", type="Place", weight=1.0)
        G.add_node("Sunnyvale", type="Place", weight=1.0)
        G.add_node("Acme Robotics", type="Organization", weight=1.0)

        # Family relations
        self.upsert_fact(G, "Alice Johnson", "spouse_of", "Mark Johnson", weight=1.0)
        self.upsert_fact(G, "Alice Johnson", "parent_of", "Bob Johnson", weight=1.0)
        self.upsert_fact(G, "Alice Johnson", "parent_of", "Ella Johnson", weight=1.0)
        self.upsert_fact(G, "Mark Johnson", "parent_of", "Bob Johnson", weight=1.0)
        self.upsert_fact(G, "Mark Johnson", "parent_of", "Ella Johnson", weight=1.0)
        self.upsert_fact(G, "Grandma Rose", "parent_of", "Alice Johnson", weight=0.8)

        # Attributes as relations
        self.upsert_fact(G, "Alice Johnson", "works_as", "Software Engineer", weight=0.9)
        self.upsert_fact(G, "Alice Johnson", "works_at", "Acme Robotics", weight=0.9)
        self.upsert_fact(G, "Johnson Family", "lives_at", "Maple Street 42", weight=0.9)
        self.upsert_fact(G, "Maple Street 42", "located_in", "Sunnyvale", weight=0.9)

        return G


    def upsert_fact(self, G: nx.DiGraph, s: str, r: str, o: str, weight: float = 0.2):
        """
        Add or update a triple (s, r, o). If the edge exists, increase its weight.
        """
        if not G.has_node(s):
            G.add_node(s, type="Entity", weight=1.0)
        if not G.has_node(o):
            G.add_node(o, type="Entity", weight=1.0)

        # If edge exists, increment weight; otherwise add with initial attributes
        if G.has_edge(s, o) and G[s][o].get("relation") == r:
            G[s][o]["weight"] = float(G[s][o].get("weight", 0.0)) + weight
        else:
            G.add_edge(s, o, relation=r, weight=weight)


    def strengthen_edges(self, G: nx.DiGraph, triples: List[Dict[str, Any]]):
        """
        Increase edge weights for the given triples list of dicts:
        {"subject": ..., "relation": ..., "object": ..., "delta": 0.2}
        """
        for t in triples:
            s = t.get("subject", "").strip()
            r = t.get("relation", "").strip()
            o = t.get("object", "").strip()
            delta = float(t.get("delta", 0.1))
            if s and r and o and G.has_edge(s, o) and G[s][o].get("relation") == r:
                G[s][o]["weight"] = float(G[s][o].get("weight", 0.0)) + delta


    def weaken_edges(self, G: nx.DiGraph, triples: List[Dict[str, Any]], remove_if_zero: bool = True):
        """
        Decrease edge weights for the given triples list of dicts:
        {"subject": ..., "relation": ..., "object": ..., "delta": 0.3}
        If weight <= 0 after decrement, remove the edge if remove_if_zero is True.
        """
        for t in triples:
            s = t.get("subject", "").strip()
            r = t.get("relation", "").strip()
            o = t.get("object", "").strip()
            delta = float(t.get("delta", 0.1))
            if s and r and o and G.has_edge(s, o) and G[s][o].get("relation") == r:
                new_w = float(G[s][o].get("weight", 0.0)) - delta
                if remove_if_zero and new_w <= 0:
                    G.remove_edge(s, o)
                else:
                    G[s][o]["weight"] = max(0.0, new_w)

    def search_graph(
        self,
        G: nx.DiGraph,
        entities: List[str],
        max_per_entity: int = 12,
        max_depth: int = 5,
        include_out: bool = True,
        include_in: bool = True,
        depth_decay: float = 0.5,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Retrieve outgoing and/or incoming facts up to max_depth hops away.
        Returns list of (subject, relation, object, score), where:
        score = edge_weight * (depth_decay ** (depth_from_entity - 1)).

        Behavior:
        - Compute shortest-path distances from each entity with cutoff=max_depth in
        the chosen direction(s), plus an undirected view for back-links/cycles. 
        - Collect ALL connecting edges among nodes whose distance <= max_depth in
        any of these views (out/in/undirected), not just BFS tree edges.
        - For an edge (u -> v), the depth used is the minimum of:
            dist_out[v] (if available), dist_in[u] (if available),
            or min(dist_undir[u], dist_undir[v]) as a fallback.
        """
        triples: List[Tuple[str, str, str, float]] = []
        ent_set: Set[str] = {e.strip() for e in entities if e and e.strip()}
        seen_edges: Set[Tuple[str, str]] = set()  # dedupe across entities/directions

        # Use an undirected view to capture “linked back” connectivity without copying
        G_undir = G.to_undirected(as_view=True)

        for ent in ent_set:
            if not G.has_node(ent):
                continue

            # Outbound distances (successors) within max_depth
            dist_out: Dict[str, int] = {}
            if include_out:
                dist_out = nx.single_source_shortest_path_length(G, ent, cutoff=max_depth)

            # Inbound distances via reversed graph within max_depth
            dist_in: Dict[str, int] = {}
            G_rev = None
            if include_in:
                G_rev = G.reverse(copy=False)
                dist_in = nx.single_source_shortest_path_length(G_rev, ent, cutoff=max_depth)

            # Undirected distances to capture back-linked/cycle nodes within max_depth
            dist_undir: Dict[str, int] = nx.single_source_shortest_path_length(
                G_undir, ent, cutoff=max_depth
            )

            # Union frontier: any node reachable in out/in/undirected spheres
            frontier_nodes: Set[str] = set(dist_undir)
            frontier_nodes.update(dist_out.keys())
            frontier_nodes.update(dist_in.keys())

            # Single count cap across all edge inclusions for this entity
            count = 0

            # Helper to compute edge depth from entity
            def edge_depth(u: str, v: str) -> int:
                candidates = []
                if include_out and v in dist_out:
                    candidates.append(dist_out[v])
                if include_in and u in dist_in:
                    candidates.append(dist_in[u])
                # Undirected fallback: distance to nearest endpoint
                du = dist_undir.get(u, math.inf)
                dv = dist_undir.get(v, math.inf)
                if du != math.inf or dv != math.inf:
                    candidates.append(min(du, dv))
                if not candidates:
                    return None
                d = max(1, int(min(candidates)))
                return d

            # Iterate all directed edges among frontier nodes (covers out, in, and cycle closures)
            for u in frontier_nodes:
                # Limit scan to successors of nodes inside the frontier
                for v in G.successors(u):
                    if v not in frontier_nodes:
                        continue
                    if (u, v) in seen_edges:
                        continue
                    data = G.get_edge_data(u, v, default={})
                    if not data:
                        continue
                    d = edge_depth(u, v)
                    if d is None or d > max_depth:
                        continue
                    rel = data.get("relation", "")
                    w = float(data.get("weight", 0.0))
                    score = w * (depth_decay ** (d - 1))
                    triples.append((u, rel, v, score))
                    seen_edges.add((u, v))
                    count += 1
                    if count >= max_per_entity:
                        break
                if count >= max_per_entity:
                    break


        # Sort by score descending for relevance
        triples.sort(key=lambda x: x[3], reverse=True)
        return triples


    def format_triples_for_context(self, triples: List[Tuple[str, str, str, float]]) -> str:
        """
        Turn triples into a compact textual context for the LLM.
        """
        lines = []
        for s, r, o, w in triples[:50]:
            lines.append(f"- ({s}) -[{r}; weight={w:.2f}]-> ({o})")
        return "\n".join(lines) if lines else "(no relevant facts found)"


    def safe_json_loads(self, s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except Exception:
            # attempt to extract JSON between fences
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start:end+1])
                except Exception:
                    pass
            return {}

    def save_graph_json(self, G: nx.DiGraph, path: str):
        data = json_graph.node_link_data(G)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


    def load_graph_json(self, path: str) -> nx.DiGraph:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        H = json_graph.node_link_graph(data, directed=True)
        # Ensure DiGraph (node_link_graph returns DiGraph when directed=True)
        if not isinstance(H, nx.DiGraph):
            H = nx.DiGraph(H)
        return H
    
    def ollama_planner(self, model: str, chat_url: str, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        self.logger.debug(f"Sending Payload: {messages}")
        client = Client(host=chat_url)
        resp = client.chat(model=model, messages=messages)
        return resp["message"]["content"]

    def strip_think_blocks_clean(self, s: str) -> str:
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
        # optional: collapse excessive blank lines created by removals
        s = re.sub(r"\n{3,}", "\n\n", s).strip()
        return s

    def plan_actions(self, G: nx.DiGraph, model: str, ollama_url: str, user_query: str) -> Dict[str, Any]:
        plan_raw = self.ollama_planner(
            model=model,
            chat_url=ollama_url,
            system=self.sys_planner,
            user=f"Knowledge Graph Overview of Edges with Attributes:\n{list(G.edges(data=True))}\n\nUser message:\n{user_query}\n\nReturn JSON plan now."
        )
        self.logger.debug(f"Generated Raw Plan: {plan_raw}")
        plan = self.safe_json_loads(self.strip_think_blocks_clean(plan_raw))
        # Ensure keys exist
        plan.setdefault("lookups", [])
        plan.setdefault("new_facts", [])
        plan.setdefault("strengthen", [])
        plan.setdefault("notes", "")
        plan.setdefault("weaken", [])
        self.logger.debug(f"Cleaned Plan: {plan}")
        return plan

    # -----------------------------
    # Agent Loop
    # -----------------------------

    def handle_user_query(self, G: nx.DiGraph, plan_model: str, ollama_plan_url: str, user_query: str) -> str:
        ollama_plan_url = "/".join(ollama_plan_url.split("/")[:3]) # Using Ollama library for interactions, no need for full URL

        # Ask LLM for a plan (what to lookup/add/strengthen)
        plan = self.plan_actions(G, plan_model, ollama_plan_url, user_query)

        # Execute plan
        lookups = [e for e in plan.get("lookups", []) if isinstance(e, str)]
        new_facts = [t for t in plan.get("new_facts", []) if isinstance(t, dict)]
        strengthen = [t for t in plan.get("strengthen", []) if isinstance(t, dict)]
        weaken = [t for t in plan.get("weaken", []) if isinstance(t, dict)]

        # Apply additions
        for t in new_facts:
            s = t.get("subject", "").strip()
            r = t.get("relation", "").strip()
            o = t.get("object", "").strip()
            w = float(t.get("weight", 0.2))
            if s and r and o:
                self.upsert_fact(G, s, r, o, weight=w)

        # Apply strengthen/weaken operations
        self.strengthen_edges(G, strengthen)
        self.weaken_edges(G, weaken)

        # Look up and return graph context
        triples = self.search_graph(G, lookups, max_per_entity=20)
        context = self.format_triples_for_context(triples)
        return context
