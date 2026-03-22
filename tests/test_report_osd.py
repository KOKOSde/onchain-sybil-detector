import json
from pathlib import Path

from datasets.synthetic_generator import generate_synthetic_sybil_network
from sybil_detector import SybilDetector, extract_features
from sybil_detector.report_osd import build_cluster_report_rows, generate_analyst_reports


def test_report_osd_generates_structured_rows_and_artifacts(tmp_path: Path) -> None:
    tx, _ = generate_synthetic_sybil_network(
        num_legit=120,
        num_sybil_clusters=6,
        addrs_per_cluster=16,
        seed=42,
    )
    clusters = SybilDetector(min_cluster_size=3, min_samples=2).fit_predict(extract_features(tx))

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

    generated = generate_analyst_reports(tx, clusters, output_dir=str(tmp_path), run_name="unit_test")

    for key in ["markdown", "json", "html", "graph"]:
        out = Path(generated[key])
        assert out.exists(), f"missing {key} output"
        assert out.stat().st_size > 0

    payload = json.loads(Path(generated["json"]).read_text(encoding="utf-8"))
    assert payload["cluster_count"] == len(payload["clusters"])
    html = Path(generated["html"]).read_text(encoding="utf-8")
    assert "Embedded Network Graph" in html
