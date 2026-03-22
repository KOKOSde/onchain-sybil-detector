"""Visualization and report generation for Sybil clusters."""

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import networkx as nx
import pandas as pd

try:
    from pyvis.network import Network  # type: ignore
except Exception:  # pragma: no cover
    Network = None


def _cluster_color(cluster_id: int) -> str:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    if cluster_id < 0:
        return "#bcbcbc"
    return palette[cluster_id % len(palette)]


def _normalize_breakdown(value: object) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))}
    if isinstance(value, str) and value.strip():
        try:
            obj = json.loads(value)
            if isinstance(obj, dict):
                return {
                    str(k): float(v)
                    for k, v in obj.items()
                    if isinstance(v, (int, float))
                }
        except Exception:
            return {}
    return {}


def plot_cluster_graph(
    transactions: pd.DataFrame,
    clusters: pd.DataFrame,
    output_html: Union[str, Path] = "cluster_graph.html",
) -> str:
    """Build an interactive transaction graph as HTML.

    Nodes are addresses, edges are transactions, color by cluster, size by sybil_probability.
    """

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph_html, _, _ = build_cluster_graph_html(transactions=transactions, clusters=clusters, cdn_resources="remote")
    output_path.write_text(graph_html, encoding="utf-8")
    return str(output_path)


def build_cluster_graph_html(
    transactions: pd.DataFrame,
    clusters: pd.DataFrame,
    cdn_resources: str = "remote",
) -> Tuple[str, int, int]:
    """Build pyvis graph HTML and return (html, node_count, edge_count)."""

    tx = transactions.copy()
    cl = clusters.copy()

    tx["from_addr"] = tx["from_addr"].astype(str).str.lower()
    tx["to_addr"] = tx["to_addr"].astype(str).str.lower()
    cl["address"] = cl["address"].astype(str).str.lower()

    cluster_map = cl.set_index("address").to_dict(orient="index")

    graph = nx.DiGraph()
    for _, row in tx.iterrows():
        src = str(row["from_addr"]).lower()
        dst = str(row["to_addr"]).lower()
        graph.add_node(src)
        graph.add_node(dst)
        graph.add_edge(src, dst, value=float(row.get("value_wei", 0)))

    cluster_prob = cl.groupby("cluster_id", as_index=False)["sybil_probability"].mean()
    flagged_clusters = set(
        cluster_prob[cluster_prob["sybil_probability"] >= 0.7]["cluster_id"].astype(int).tolist()
    )

    node_count = int(graph.number_of_nodes())
    edge_count = int(graph.number_of_edges())

    if Network is None:
        fallback = [
            "<html><body><h3>Cluster Graph (fallback)</h3>",
            "<p>pyvis is not installed, rendering textual summary instead.</p>",
            f"<p>nodes={node_count}, edges={edge_count}</p>",
            "</body></html>",
        ]
        return "".join(fallback), node_count, edge_count

    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        cdn_resources=cdn_resources,
    )

    for node in graph.nodes:
        node_info: Dict[str, object] = cluster_map.get(node, {})
        cluster_id = int(node_info.get("cluster_id", -1))
        sybil_prob = float(node_info.get("sybil_probability", 0.05))
        color = _cluster_color(cluster_id)

        border = "#ff2d2d" if cluster_id in flagged_clusters else "#2f2f2f"
        size = 12 + int(30 * sybil_prob)

        net.add_node(
            node,
            label=node[:10] + "...",
            title="{}<br>cluster={}<br>p_sybil={:.2f}".format(node, cluster_id, sybil_prob),
            color={"background": color, "border": border, "highlight": {"background": color, "border": border}},
            size=size,
        )

    for src, dst, attrs in graph.edges(data=True):
        net.add_edge(src, dst, value=float(attrs.get("value", 0.0)), color="#9aa0a6")

    net.toggle_physics(True)
    return net.generate_html(notebook=False), node_count, edge_count


def generate_report(clusters: pd.DataFrame, format: str = "html") -> str:
    """Generate per-cluster evidence report with timeline hints and actions."""

    cl = clusters.copy()
    if cl.empty:
        return "<p>No clusters found.</p>" if format == "html" else "No clusters found."

    cl["cluster_id"] = cl["cluster_id"].astype(int)
    grouped = cl[cl["cluster_id"] != -1].groupby("cluster_id", as_index=False).agg(
        size=("address", "count"),
        mean_sybil_probability=("sybil_probability", "mean"),
    )

    lines = []
    for _, row in grouped.sort_values("mean_sybil_probability", ascending=False).iterrows():
        cid = int(row["cluster_id"])
        size = int(row["size"])
        prob = float(row["mean_sybil_probability"])

        sample = cl[cl["cluster_id"] == cid].iloc[0]
        breakdown = _normalize_breakdown(sample.get("coordination_breakdown", {}))
        top_feats = sample.get("top_3_features", [])
        if isinstance(top_feats, list):
            top_feats_text = ", ".join([str(x) for x in top_feats])
        else:
            top_feats_text = str(top_feats)

        if prob >= 0.8:
            action = "Block or quarantine pending manual review"
        elif prob >= 0.6:
            action = "Flag for secondary risk scoring"
        else:
            action = "Monitor only"

        lines.append(
            {
                "cluster_id": cid,
                "size": size,
                "prob": prob,
                "breakdown": breakdown,
                "top_feats": top_feats_text,
                "action": action,
            }
        )

    if format == "html":
        parts = [
            "<html><head><meta charset='utf-8'><title>Sybil Cluster Report</title></head><body>",
            "<h1>Sybil Coordination Report</h1>",
            "<p>Generated from cluster-level behavioral coordination signals.</p>",
            "<ul>",
        ]
        for item in lines:
            parts.append(
                "<li>"
                "<b>Cluster {}</b> | size={} | sybil_probability={:.2f}<br>"
                "Top features: {}<br>"
                "Coordination breakdown: {}<br>"
                "Recommended action: <b>{}</b>"
                "</li><br>".format(
                    item["cluster_id"],
                    item["size"],
                    item["prob"],
                    item["top_feats"],
                    item["breakdown"],
                    item["action"],
                )
            )
        parts.extend(["</ul>", "</body></html>"])
        return "".join(parts)

    text_lines = ["Sybil Coordination Report"]
    for item in lines:
        text_lines.append(
            "Cluster {} | size={} | p={:.2f} | top={} | breakdown={} | action={}".format(
                item["cluster_id"],
                item["size"],
                item["prob"],
                item["top_feats"],
                item["breakdown"],
                item["action"],
            )
        )
    return "\n".join(text_lines)


__all__ = ["plot_cluster_graph", "build_cluster_graph_html", "generate_report"]
