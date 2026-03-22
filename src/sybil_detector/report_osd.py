"""Analyst report generation for onchain-sybil-detector runs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .visualization import plot_cluster_graph


def _ts_to_iso(ts: object) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _safe_json_obj(value: object) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if isinstance(v, (float, int))}
    if isinstance(value, str) and value.strip():
        try:
            obj = json.loads(value)
            if isinstance(obj, dict):
                return {str(k): float(v) for k, v in obj.items() if isinstance(v, (float, int))}
        except Exception:
            return {}
    return {}


def _mean_coordination_breakdown(cluster_group: pd.DataFrame) -> Dict[str, float]:
    merged: Dict[str, List[float]] = {}
    if "coordination_breakdown" not in cluster_group.columns:
        return {}

    for raw in cluster_group["coordination_breakdown"].tolist():
        obj = _safe_json_obj(raw)
        for key, score in obj.items():
            merged.setdefault(str(key), []).append(float(score))

    return {k: float(np.mean(v)) for k, v in merged.items() if v}


def _cluster_transactions(transactions: pd.DataFrame, addresses: List[str]) -> pd.DataFrame:
    tx = transactions.copy()
    for col in ["address", "from_addr", "to_addr"]:
        if col in tx.columns:
            tx[col] = tx[col].astype(str).str.lower()

    addr_set = set(addresses)
    mask = pd.Series(False, index=tx.index)
    for col in ["address", "from_addr", "to_addr"]:
        if col in tx.columns:
            mask = mask | tx[col].isin(addr_set)

    subset = tx.loc[mask].copy()
    if "tx_hash" in subset.columns:
        dedupe_cols = ["tx_hash"]
        if "address" in subset.columns:
            dedupe_cols.append("address")
        subset = subset.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
    return subset


def _timeline_summary(cluster_tx: pd.DataFrame) -> Dict[str, object]:
    if cluster_tx.empty or "timestamp" not in cluster_tx.columns:
        return {
            "first_tx": None,
            "peak_activity": None,
            "last_tx": None,
            "peak_hour_count": 0,
        }

    ts = pd.to_numeric(cluster_tx["timestamp"], errors="coerce").dropna().astype(np.int64)
    if ts.empty:
        return {
            "first_tx": None,
            "peak_activity": None,
            "last_tx": None,
            "peak_hour_count": 0,
        }

    dt = pd.to_datetime(ts, unit="s", utc=True)
    hourly = dt.dt.floor("h").value_counts().sort_index()
    peak_hour = hourly.idxmax() if len(hourly) else None

    return {
        "first_tx": _ts_to_iso(int(ts.min())),
        "peak_activity": peak_hour.isoformat() if peak_hour is not None else None,
        "last_tx": _ts_to_iso(int(ts.max())),
        "peak_hour_count": int(hourly.max()) if len(hourly) else 0,
    }


def _funding_graph_summary(cluster_tx: pd.DataFrame, addresses: List[str]) -> Dict[str, object]:
    if cluster_tx.empty or "from_addr" not in cluster_tx.columns or "to_addr" not in cluster_tx.columns:
        return {
            "external_funding_edges": 0,
            "top_funders": [],
            "common_ancestor_candidates": [],
            "edges": 0,
        }

    addr_set = set(addresses)
    tx = cluster_tx.copy()
    incoming = tx[(tx["to_addr"].isin(addr_set)) & (~tx["from_addr"].isin(addr_set))]

    top_funders = (
        incoming["from_addr"]
        .value_counts()
        .head(5)
        .rename_axis("wallet")
        .reset_index(name="funded_wallet_events")
    )

    graph = nx.DiGraph()
    for _, row in tx.iterrows():
        graph.add_edge(str(row["from_addr"]), str(row["to_addr"]))

    ancestor_candidates = []
    for node in graph.nodes:
        outd = graph.out_degree(node)
        if outd >= max(2, int(len(addresses) * 0.25)):
            ancestor_candidates.append((str(node), int(outd)))

    ancestor_candidates = sorted(ancestor_candidates, key=lambda x: x[1], reverse=True)[:5]

    return {
        "external_funding_edges": int(len(incoming[["from_addr", "to_addr"]].drop_duplicates())),
        "top_funders": top_funders.to_dict(orient="records"),
        "common_ancestor_candidates": [
            {"wallet": wallet, "out_degree": out_degree}
            for wallet, out_degree in ancestor_candidates
        ],
        "edges": int(graph.number_of_edges()),
    }


def _gas_summary(cluster_tx: pd.DataFrame) -> Dict[str, float]:
    if cluster_tx.empty:
        return {
            "median_gas_price": 0.0,
            "gas_price_cv": 0.0,
            "median_gas_used": 0.0,
            "strategy_similarity": 0.0,
        }

    if "gas_price" in cluster_tx.columns:
        gas_price = pd.to_numeric(cluster_tx["gas_price"], errors="coerce").fillna(0.0).astype(float)
    else:
        gas_price = pd.Series([0.0] * len(cluster_tx), dtype=float)

    if "gas_used" in cluster_tx.columns:
        gas_used = pd.to_numeric(cluster_tx["gas_used"], errors="coerce").fillna(0.0).astype(float)
    else:
        gas_used = pd.Series([0.0] * len(cluster_tx), dtype=float)

    cv = float(gas_price.std(ddof=0) / gas_price.mean()) if gas_price.mean() > 0 else 0.0
    return {
        "median_gas_price": float(gas_price.median()),
        "gas_price_cv": cv,
        "median_gas_used": float(gas_used.median()),
        "strategy_similarity": float(np.clip(np.exp(-8.0 * cv), 0.0, 1.0)),
    }


def _suspicious_subgroups(cluster_tx: pd.DataFrame, addresses: List[str]) -> List[Dict[str, object]]:
    if cluster_tx.empty or "from_addr" not in cluster_tx.columns or "to_addr" not in cluster_tx.columns:
        return []

    addr_set = set(addresses)
    incoming = cluster_tx[(cluster_tx["to_addr"].isin(addr_set)) & (~cluster_tx["from_addr"].isin(addr_set))]
    groups = []

    for funder, grp in incoming.groupby("from_addr"):
        funded = sorted(grp["to_addr"].astype(str).str.lower().unique().tolist())
        if len(funded) < 2:
            continue

        if "timestamp" in grp.columns:
            ts = pd.to_numeric(grp["timestamp"], errors="coerce").dropna()
            span = float((ts.max() - ts.min()) / 60.0) if not ts.empty else 0.0
        else:
            span = 0.0

        groups.append(
            {
                "group_type": "shared_funder",
                "anchor_wallet": str(funder),
                "member_wallets": funded,
                "size": int(len(funded)),
                "funding_window_minutes": float(max(span, 0.0)),
            }
        )

    groups = sorted(groups, key=lambda x: int(x["size"]), reverse=True)
    return groups[:5]


def _strongest_evidence(coord: Dict[str, float]) -> List[Dict[str, object]]:
    ranked = sorted(coord.items(), key=lambda x: float(x[1]), reverse=True)
    return [{"evidence_type": key, "score": float(score)} for key, score in ranked]


def _summary_sentence(
    cluster_id: int,
    addresses: List[str],
    confidence: float,
    timeline: Dict[str, object],
    funding: Dict[str, object],
    gas: Dict[str, float],
) -> str:
    top_funders = funding.get("top_funders", [])
    funder_text = top_funders[0]["wallet"] if top_funders else "unknown source"

    window_minutes = 0.0
    if timeline.get("first_tx") and timeline.get("last_tx"):
        t0 = pd.Timestamp(timeline["first_tx"])
        t1 = pd.Timestamp(timeline["last_tx"])
        window_minutes = (t1 - t0).total_seconds() / 60.0

    return (
        "Cluster {cid}: {count} wallets funded by {funder} within {window:.1f} minutes, "
        "confidence={conf:.2f}, gas CV={gas_cv:.3f}."
    ).format(
        cid=cluster_id,
        count=len(addresses),
        funder=funder_text,
        window=max(window_minutes, 0.0),
        conf=confidence,
        gas_cv=float(gas.get("gas_price_cv", 0.0)),
    )


def build_cluster_report_rows(
    transactions: pd.DataFrame,
    clusters: pd.DataFrame,
) -> List[Dict[str, object]]:
    """Build report rows for each non-noise cluster."""

    cl = clusters.copy()
    cl["address"] = cl["address"].astype(str).str.lower()
    cl["cluster_id"] = pd.to_numeric(cl["cluster_id"], errors="coerce").fillna(-1).astype(int)
    cl["sybil_probability"] = pd.to_numeric(cl["sybil_probability"], errors="coerce").fillna(0.0)

    rows: List[Dict[str, object]] = []
    for cluster_id, grp in cl[cl["cluster_id"] != -1].groupby("cluster_id"):
        addresses = sorted(grp["address"].unique().tolist())
        confidence = float(np.clip(grp["sybil_probability"].mean(), 0.0, 1.0))

        coord = _mean_coordination_breakdown(grp)
        strongest = _strongest_evidence(coord)

        c_tx = _cluster_transactions(transactions, addresses)
        timeline = _timeline_summary(c_tx)
        funding = _funding_graph_summary(c_tx, addresses)
        gas = _gas_summary(c_tx)
        subgroups = _suspicious_subgroups(c_tx, addresses)
        summary = _summary_sentence(int(cluster_id), addresses, confidence, timeline, funding, gas)

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "wallet_count": int(len(addresses)),
                "confidence_score": confidence,
                "strongest_evidence_types": strongest,
                "timeline": timeline,
                "funding_graph_summary": funding,
                "gas_pattern_similarity_summary": gas,
                "top_suspicious_subgroups": subgroups,
                "summary_sentence": summary,
                "addresses": addresses,
            }
        )

    rows = sorted(rows, key=lambda x: float(x["confidence_score"]), reverse=True)
    return rows


def generate_analyst_reports(
    transactions: pd.DataFrame,
    clusters: pd.DataFrame,
    output_dir: str = "reports",
    run_name: Optional[str] = None,
) -> Dict[str, str]:
    """Generate markdown/json/html analyst reports with embedded network graph."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = run_name or datetime.now(timezone.utc).strftime("osd_report_%Y%m%d_%H%M%S")
    rows = build_cluster_report_rows(transactions=transactions, clusters=clusters)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cluster_count": len(rows),
        "clusters": rows,
    }

    json_path = out_dir / f"{stem}_analyst_report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        f"# Analyst Report ({stem})",
        "",
        f"Generated at: {payload['generated_at']}",
        f"Clusters analyzed: {payload['cluster_count']}",
        "",
    ]
    for row in rows:
        md_lines.append(f"## Cluster {row['cluster_id']}")
        md_lines.append(f"- Wallet count: {row['wallet_count']}")
        md_lines.append(f"- Confidence score: {row['confidence_score']:.3f}")
        md_lines.append(f"- Timeline: {row['timeline']}")
        md_lines.append(f"- Funding summary: {row['funding_graph_summary']}")
        md_lines.append(f"- Gas summary: {row['gas_pattern_similarity_summary']}")
        md_lines.append(f"- Strongest evidence: {row['strongest_evidence_types']}")
        md_lines.append(f"- Suspicious subgroups: {row['top_suspicious_subgroups']}")
        md_lines.append(f"- Summary: {row['summary_sentence']}")
        md_lines.append("")

    md_path = out_dir / f"{stem}_analyst_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    graph_path = out_dir / f"{stem}_graph.html"
    try:
        plot_cluster_graph(transactions=transactions, clusters=clusters, output_html=graph_path)
    except Exception as exc:
        graph_path.write_text(
            (
                "<html><body>"
                "<h3>Graph unavailable</h3>"
                f"<p>Unable to render pyvis graph: {exc}</p>"
                "</body></html>"
            ),
            encoding="utf-8",
        )

    blocks = [
        "<html><head><meta charset='utf-8'><title>OSD Analyst Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:8px;vertical-align:top;}"
        "th{background:#f7f7f7;}"
        "iframe{width:100%;height:760px;border:1px solid #ddd;}</style></head><body>",
        f"<h1>Analyst Report ({stem})</h1>",
        f"<p>Generated at {payload['generated_at']}</p>",
        f"<p>Clusters analyzed: {payload['cluster_count']}</p>",
        "<h2>Cluster Summaries</h2>",
        "<table><thead><tr><th>Cluster</th><th>Wallets</th><th>Confidence</th><th>Summary</th></tr></thead><tbody>",
    ]

    for row in rows:
        blocks.append(
            "<tr>"
            f"<td>{row['cluster_id']}</td>"
            f"<td>{row['wallet_count']}</td>"
            f"<td>{row['confidence_score']:.3f}</td>"
            f"<td>{row['summary_sentence']}</td>"
            "</tr>"
        )

    blocks.extend(
        [
            "</tbody></table>",
            "<h2>Embedded Network Graph</h2>",
            f"<iframe src='{graph_path.name}'></iframe>",
            "</body></html>",
        ]
    )

    html_path = out_dir / f"{stem}_analyst_report.html"
    html_path.write_text("".join(blocks), encoding="utf-8")

    return {
        "markdown": str(md_path),
        "json": str(json_path),
        "html": str(html_path),
        "graph": str(graph_path),
    }


__all__ = ["build_cluster_report_rows", "generate_analyst_reports"]
