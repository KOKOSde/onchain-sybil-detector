import json
import re
from pathlib import Path

from sybil_detector import SybilDetector, extract_features
from sybil_detector.datasets.synthetic_generator import generate_synthetic_sybil_network
from sybil_detector.report_osd import build_cluster_report_rows, generate_analyst_reports, generate_report


def _generate(tmp_path: Path) -> tuple:
    tx, labels = generate_synthetic_sybil_network(
        num_legit=120,
        num_sybil_clusters=6,
        addrs_per_cluster=16,
        seed=42,
    )
    features = extract_features(tx)
    clusters = SybilDetector(min_cluster_size=3, min_samples=2).fit_predict(features)
    generated = generate_analyst_reports(tx, clusters, output_dir=str(tmp_path), run_name="unit_test")
    return tx, labels, features, clusters, generated


def test_report_osd_generates_structured_rows_and_artifacts(tmp_path: Path) -> None:
    tx, _, _, clusters, generated = _generate(tmp_path)

    rows = build_cluster_report_rows(tx, clusters)
    assert isinstance(rows, list)
    for row in rows:
        assert {
            "cluster_id",
            "wallet_count",
            "confidence_score",
            "strongest_evidence_types",
            "timeline",
            "funding_graph_summary",
            "gas_pattern_similarity_summary",
            "top_suspicious_subgroups",
            "summary_sentence",
            "addresses",
        }.issubset(set(row.keys()))

    for key in ["markdown", "json", "html", "graph"]:
        out = Path(generated[key])
        assert out.exists(), f"missing {key} output"
        assert out.stat().st_size > 0


def test_report_osd_json_is_valid_and_has_expected_keys(tmp_path: Path) -> None:
    _, _, _, _, generated = _generate(tmp_path)
    payload = json.loads(Path(generated["json"]).read_text(encoding="utf-8"))

    assert set(payload.keys()) == {"generated_at", "cluster_count", "clusters"}
    assert payload["cluster_count"] == len(payload["clusters"])
    if payload["clusters"]:
        first = payload["clusters"][0]
        assert {
            "cluster_id",
            "wallet_count",
            "confidence_score",
            "strongest_evidence_types",
            "timeline",
            "funding_graph_summary",
            "gas_pattern_similarity_summary",
            "top_suspicious_subgroups",
            "summary_sentence",
            "addresses",
        }.issubset(set(first.keys()))


def test_report_osd_markdown_contains_expected_headers(tmp_path: Path) -> None:
    _, _, _, _, generated = _generate(tmp_path)
    md = Path(generated["markdown"]).read_text(encoding="utf-8")
    assert md.startswith("# Analyst Report")
    assert "## Cluster" in md
    assert "- Wallet count:" in md
    assert "- Summary:" in md


def test_report_osd_html_contains_non_empty_embedded_graph(tmp_path: Path) -> None:
    _, _, _, _, generated = _generate(tmp_path)
    html = Path(generated["html"]).read_text(encoding="utf-8")
    graph_html = Path(generated["graph"]).read_text(encoding="utf-8")

    assert "Embedded Network Graph" in html
    assert "srcdoc=" in html
    assert "Graph stats: nodes=" in html

    m = re.search(r"Graph stats: nodes=(\d+), edges=(\d+)", html)
    assert m is not None
    assert int(m.group(1)) > 0
    assert int(m.group(2)) > 0

    assert "Graph unavailable" not in graph_html
    assert ("vis-network" in graph_html) or ("vis.Network" in graph_html)


def test_report_osd_generate_report_wrapper_compat_signature(tmp_path: Path) -> None:
    tx, labels = generate_synthetic_sybil_network(num_legit=60, num_sybil_clusters=3, addrs_per_cluster=8, seed=7)
    features = extract_features(tx)
    clusters = SybilDetector(min_cluster_size=3, min_samples=2).fit_predict(features)

    outputs = generate_report(
        transactions=tx,
        features=features,
        clusters=clusters,
        labels=labels,
        output_dir=str(tmp_path),
        filename_prefix="compat_test",
    )
    assert Path(outputs["html"]).exists()
    assert Path(outputs["json"]).exists()
