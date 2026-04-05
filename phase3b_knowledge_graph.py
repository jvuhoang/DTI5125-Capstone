"""
Phase 3B — Biomedical Knowledge Graph (NetworkX + Gephi + D3.js)
================================================================
Builds an undirected weighted co-occurrence graph from NER entities
and PICOS intervention fields extracted in Phase 2.

Nodes  : disease labels, DISEASE entities, CHEMICAL entities, PICOS interventions
Edges  : co-occurrence within the same abstract (weight = frequency)
Pruning: edges with weight < 3 are removed to reduce noise

Outputs:
  knowledge_graph.png       — inline matplotlib visualisation
  knowledge_graph.gexf      — Gephi-compatible export (ForceAtlas2 + Modularity)
  knowledge_graph_data.json — D3.js JSON for the NORA web frontend

Run:
    python phase3b_knowledge_graph.py
    python phase3b_knowledge_graph.py --min-weight 5   # stricter pruning
"""

import argparse
import json
import sqlite3
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

DB_PATH    = "abstracts.db"
MIN_WEIGHT = 3   # minimum edge weight to keep


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(db_path: str = DB_PATH, min_weight: int = MIN_WEIGHT) -> nx.Graph:
    """
    Pull NER entities and PICOS interventions from the database.
    Build a weighted co-occurrence graph.
    """
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()
    rows = c.execute("""
        SELECT id, disease_label, ner_entities,
               picos_intervention, picos_outcome
        FROM abstracts
        WHERE ner_entities IS NOT NULL
    """).fetchall()
    conn.close()

    print(f"Building graph from {len(rows)} annotated abstracts...")

    G            = nx.Graph()
    edge_weights = defaultdict(int)

    for row_id, disease_label, ner_json, intervention, outcome in rows:
        try:
            entities = json.loads(ner_json) if ner_json else []
        except json.JSONDecodeError:
            entities = []

        node_set = set()

        # Anchor node — disease label (from PubMed query label)
        if disease_label:
            anchor = disease_label.lower().strip()
            G.add_node(anchor, node_type="disease", size=20)
            node_set.add(anchor)

        # NER entities — DISEASE and CHEMICAL
        for ent in entities:
            text  = ent.get("text", "").lower().strip()
            label = ent.get("label", "")
            if len(text) < 3 or len(text) > 50:
                continue
            ntype = "disease" if label == "DISEASE" else "chemical"
            if not G.has_node(text):
                G.add_node(text, node_type=ntype, size=5)
            node_set.add(text)

        # PICOS intervention
        if intervention and intervention.lower() not in ("not reported", "", "extraction_failed"):
            interv = intervention.lower().strip()[:60]
            if len(interv) >= 3:
                if not G.has_node(interv):
                    G.add_node(interv, node_type="intervention", size=8)
                node_set.add(interv)

        # Accumulate co-occurrence edges
        node_list = list(node_set)
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                a, b = node_list[i], node_list[j]
                edge_weights[(a, b)] += 1

    # Add edges that meet the weight threshold
    for (u, v), w in edge_weights.items():
        if w >= min_weight and G.has_node(u) and G.has_node(v):
            G.add_edge(u, v, weight=w)

    # Remove isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"  Removed {len(isolates)} isolated nodes")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


# ── Visualisation ─────────────────────────────────────────────────────────────

COLOR_MAP = {
    "disease":      "#E63946",
    "chemical":     "#457B9D",
    "intervention": "#2A9D8F",
}


def visualise(G: nx.Graph, output: str = "knowledge_graph.png") -> None:
    """Render the graph inline using matplotlib spring layout."""
    print("Rendering graph visualisation...")

    node_colors = [
        COLOR_MAP.get(G.nodes[n].get("node_type", "disease"), "#888888")
        for n in G.nodes()
    ]
    node_sizes = [300 + G.degree(n) * 80 for n in G.nodes()]
    edge_widths = [G[u][v].get("weight", 1) * 0.3 for u, v in G.edges()]

    plt.figure(figsize=(18, 13))
    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=60)

    nx.draw_networkx_edges(G, pos,
                           width=edge_widths, alpha=0.25,
                           edge_color="gray")
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.85)

    # Label only the highest-degree nodes to avoid clutter
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top_n   = min(50, len(degrees))
    degree_threshold = degrees[top_n - 1][1]
    labels_to_draw = {n: n for n in G.nodes() if G.degree(n) >= degree_threshold}
    nx.draw_networkx_labels(G, pos, labels=labels_to_draw,
                            font_size=7, font_weight="bold")

    patches = [
        mpatches.Patch(color=c, label=t.capitalize())
        for t, c in COLOR_MAP.items()
    ]
    plt.legend(handles=patches, loc="upper left", fontsize=11)
    plt.title(
        "Biomedical Knowledge Graph — 6-disease Neurodegenerative Corpus\n"
        "(nodes: diseases · chemicals · interventions — edges: co-occurrence weight)",
        fontsize=13,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


# ── Gephi export ──────────────────────────────────────────────────────────────

GEPHI_COLORS = {
    "disease":      (230, 57,  70),
    "chemical":     (69,  123, 157),
    "intervention": (42,  157, 143),
}


def export_gephi(G: nx.Graph, output: str = "knowledge_graph.gexf") -> None:
    """Export graph as .gexf for Gephi with node colour attributes."""
    for node in G.nodes():
        ntype = G.nodes[node].get("node_type", "disease")
        r, g, b = GEPHI_COLORS.get(ntype, (128, 128, 128))
        G.nodes[node]["label"] = node
        G.nodes[node]["r"]     = r
        G.nodes[node]["g"]     = g
        G.nodes[node]["b"]     = b

    nx.write_gexf(G, output)
    print(f"Saved: {output}")
    print("  → Open in Gephi, run ForceAtlas2 layout, apply Modularity colouring")
    print("    to reveal disease community clusters.")


# ── D3.js JSON export ─────────────────────────────────────────────────────────

def export_d3_json(G: nx.Graph, output: str = "knowledge_graph_data.json") -> None:
    """Export graph as JSON for the D3.js force-directed graph in index.html."""
    graph_data = {
        "nodes": [
            {
                "id":        node,
                "node_type": G.nodes[node].get("node_type", "disease"),
                "degree":    G.degree(node),
            }
            for node in G.nodes()
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "weight": G[u][v].get("weight", 1),
            }
            for u, v in G.edges()
        ],
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output}")
    print(f"  → {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
    print("  → Place alongside index.html — the Knowledge Graph tab loads it automatically.")


# ── Graph statistics ──────────────────────────────────────────────────────────

def print_statistics(G: nx.Graph) -> None:
    """Print key graph statistics for the written report."""
    print(f"\n{'─'*50}")
    print("Graph Statistics")
    print(f"  Nodes:          {G.number_of_nodes()}")
    print(f"  Edges:          {G.number_of_edges()}")
    print(f"  Density:        {nx.density(G):.4f}")
    avg_deg = sum(d for _, d in G.degree()) / G.number_of_nodes()
    print(f"  Average degree: {avg_deg:.2f}")

    dc       = nx.degree_centrality(G)
    top_nodes = sorted(dc.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 most connected nodes:")
    for name, centrality in top_nodes:
        ntype = G.nodes[name].get("node_type", "?")
        print(f"  [{ntype:12s}] {name:<40s}  centrality={centrality:.4f}")

    # Node type breakdown
    type_counts = defaultdict(int)
    for n in G.nodes():
        type_counts[G.nodes[n].get("node_type", "unknown")] += 1
    print(f"\nNode type breakdown:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t:<14}: {count}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3B: Knowledge Graph")
    parser.add_argument("--db",         default=DB_PATH,  help="Path to abstracts.db")
    parser.add_argument("--min-weight", type=int, default=MIN_WEIGHT,
                        help=f"Minimum edge weight to keep (default: {MIN_WEIGHT})")
    parser.add_argument("--no-plot",    action="store_true",
                        help="Skip matplotlib rendering (faster on headless servers)")
    args = parser.parse_args()

    G = build_graph(args.db, args.min_weight)

    if not args.no_plot:
        visualise(G)

    export_gephi(G)
    export_d3_json(G)
    print_statistics(G)

    print("\nPhase 3B complete.")
