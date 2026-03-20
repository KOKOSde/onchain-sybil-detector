from datasets.synthetic_generator import generate_synthetic_sybil_network
from sybil_detector import SybilDetector, run_benchmark


def test_benchmark_output_has_expected_keys() -> None:
    tx, labels = generate_synthetic_sybil_network(
        num_legit=200,
        num_sybil_clusters=7,
        addrs_per_cluster=15,
        seed=101,
    )
    detector = SybilDetector(min_cluster_size=3, min_samples=2)
    result = run_benchmark(detector, tx, labels)

    assert set(result.keys()) == {
        "address",
        "cluster",
        "baselines",
        "decision_threshold",
        "n_addresses",
        "n_predicted_sybil_addresses",
        "n_predicted_clusters",
    }
    assert set(result["address"].keys()) == {"precision", "recall", "f1"}
    assert set(result["cluster"].keys()) == {"precision", "recall", "f1"}
    assert set(result["baselines"].keys()) == {"same_funder", "same_gas", "same_timing"}
    assert result["address"]["precision"] >= 0.89
