"""Minimal end-to-end offline quickstart."""

from datasets.synthetic_generator import generate_synthetic_sybil_network
from sybil_detector import SybilDetector, extract_features, run_benchmark


def main() -> None:
    tx, labels = generate_synthetic_sybil_network(seed=42)
    features = extract_features(tx)
    detector = SybilDetector(min_cluster_size=3, min_samples=2)
    clusters = detector.fit_predict(features)
    metrics = run_benchmark(detector, tx, labels)
    flagged = clusters[clusters["sybil_probability"] >= metrics["decision_threshold"]]
    print("flagged_addresses=", len(flagged), "address_precision=", round(metrics["address"]["precision"], 3))


if __name__ == "__main__":
    main()
