"""
Handler for centralized AI knowledge graph.

This module provides all knowledge graph functionality 
"""
import re
import json
from typing import List, Dict, Any, Tuple, Set
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph
from ollama import Client

from config import config


class KnowledgeHandler:
    """Centralized knowledge handler"""
    
    def __init__(self):
        self.logger = config.get_logger("KGLLM")
        self.sys_planner = """You are a planning assistant that connects to a knowledge graph (KG).
Return ONLY a JSON object with keys: lookups, new_facts, strengthen, weaken, and notes.
- lookups: list of entities to fetch from the KG, e.g. ["Alice Johnson","Bob Johnson"].
- new_facts: list of triples to add if implied by the latest user message. Each: {"subject": str, "relation": str, "object": str, "weight": float}.
- strengthen: list of triples to upweight if repeated/corroborated/newer. Each: {"subject": str, "relation": str, "object": str, "delta": float}.
- weaken: list of triples to downweight if contradicted/obsolete/older. Each: {"subject": str, "relation": str, "object": str, "delta": float}.
- notes: 1-2 short bullets on your reasoning (kept brief).

Only include facts supported or directly implied by the user message.
Use common relations: parent_of, spouse_of, sibling_of, lives_at, located_in, works_as, works_at.
When a message says something like "no longer", "not anymore", prefer weaken for affected relations.
"""

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
        max_depth: int = 3,
        include_out: bool = True,
        include_in: bool = True,
        depth_decay: float = 1.0,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Retrieve outgoing and/or incoming facts up to max_depth hops away via BFS.
        Returns list of (subject, relation, object, score), where:
        score = edge_weight * (depth_decay ** (depth_from_entity - 1)).
        """
        triples: List[Tuple[str, str, str, float]] = []
        ent_set: Set[str] = set(e.strip() for e in entities if e and e.strip())
        seen_edges: Set[Tuple[str, str]] = set()  # dedupe across entities and directions

        for ent in ent_set:
            if not G.has_node(ent):
                continue

            # Precompute hop distances from ent in each direction up to cutoff
            dist_out = {}
            dist_in = {}
            if include_out:
                # Outbound distances (successors) within max_depth
                dist_out = nx.single_source_shortest_path_length(G, ent, cutoff=max_depth)
            if include_in:
                # Inbound distances (predecessors) via reversed graph within max_depth
                dist_in = nx.single_source_shortest_path_length(G.reverse(), ent, cutoff=max_depth)

            # Collect outgoing edges up to depth using BFS tree edges
            if include_out and max_per_entity > 0:
                count = 0
                for u, v in nx.bfs_edges(G, source=ent, depth_limit=max_depth):
                    if (u, v) in seen_edges:
                        continue
                    data = G.get_edge_data(u, v, default={})
                    rel = data.get("relation", "")
                    w = float(data.get("weight", 0.0))
                    depth = dist_out.get(v, 1)
                    score = w * (depth_decay ** (depth - 1))
                    triples.append((u, rel, v, score))
                    seen_edges.add((u, v))
                    count += 1
                    if count >= max_per_entity:
                        break

            # Collect incoming edges up to depth using BFS over reversed orientation
            if include_in and max_per_entity > 0:
                count = 0
                for u_rev, v_rev in nx.bfs_edges(G, source=ent, depth_limit=max_depth, reverse=True):
                    # In reversed traversal, edge (u_rev -> v_rev) corresponds to original (v_rev -> u_rev)
                    u, v = v_rev, u_rev
                    if (u, v) in seen_edges:
                        continue
                    data = G.get_edge_data(u, v, default={})
                    if not data:
                        continue  # safety: skip if edge missing in original orientation
                    rel = data.get("relation", "")
                    w = float(data.get("weight", 0.0))
                    depth = dist_in.get(v_rev, 1)
                    score = w * (depth_decay ** (depth - 1))
                    triples.append((u, rel, v, score))
                    seen_edges.add((u, v))
                    count += 1
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

    def plan_actions(self, model: str, ollama_url: str, user_query: str) -> Dict[str, Any]:
        plan_raw = self.ollama_planner(
            model=model,
            chat_url=ollama_url,
            system=self.sys_planner,
            user=f"User message:\n{user_query}\n\nReturn JSON plan now."
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

        # 1) Ask LLM for a plan (what to lookup/add/strengthen)
        plan = self.plan_actions(plan_model, ollama_plan_url, user_query)

        # 2) Execute plan
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

        # 3) Look up and return graph context
        triples = self.search_graph(G, lookups, max_per_entity=20)
        context = self.format_triples_for_context(triples)
        return context
