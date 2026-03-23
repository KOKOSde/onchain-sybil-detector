from sklearn.metrics import precision_recall_fscore_support

from sybil_detector.datasets.synthetic_generator import generate_synthetic_sybil_network
from sybil_detector import SybilDetector, extract_features


def test_clustering_finds_known_sybil_groups_with_high_f1() -> None:
    tx, labels = generate_synthetic_sybil_network(
        num_legit=220,
        num_sybil_clusters=8,
        addrs_per_cluster=16,
        seed=42,
    )
    features = extract_features(tx)
    detector = SybilDetector(min_cluster_size=3, min_samples=2)
    clusters = detector.fit_predict(features)

    merged = features[["address"]].merge(labels[["address", "is_sybil"]], on="address", how="left")
    merged = merged.merge(clusters[["address", "cluster_id", "sybil_probability"]], on="address", how="left")
    merged["is_sybil"] = merged["is_sybil"].fillna(0).astype(int)

    y_true = merged["is_sybil"].to_numpy()
    y_pred = (
        (merged["cluster_id"].fillna(-1).astype(int).to_numpy() != -1)
        & (merged["sybil_probability"].fillna(0.0).to_numpy() >= 0.70)
    ).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    assert precision >= 0.89
    assert f1 > 0.85
    assert recall > 0.70
