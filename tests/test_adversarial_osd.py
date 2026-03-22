from datasets.adversarial_simulator_osd import (
    generate_adversarial_sybils,
    run_adversarial_detection_benchmark,
)
from sybil_detector import SybilDetector


def test_generate_adversarial_levels_have_expected_schema() -> None:
    required_tx_cols = {
        "address",
        "tx_hash",
        "block_number",
        "timestamp",
        "from_addr",
        "to_addr",
        "value_wei",
        "gas_price",
        "gas_used",
        "input_data_prefix",
        "chain",
    }
    required_label_cols = {"address", "is_sybil", "cluster_id", "operator_id", "difficulty_level", "difficulty_tags"}

    for level in range(1, 9):
        tx, labels = generate_adversarial_sybils(
            difficulty=level,
            num_clusters=3,
            wallets_per_cluster=6,
            num_legit=60,
            seed=42,
        )
        assert not tx.empty
        assert not labels.empty
        assert required_tx_cols.issubset(set(tx.columns))
        assert required_label_cols.issubset(set(labels.columns))
        assert labels["difficulty_level"].max() == level


def test_enabled_levels_toggle_applies_multiple_difficulties() -> None:
    tx, labels = generate_adversarial_sybils(
        difficulty=1,
        enabled_levels=[2, 6],
        num_clusters=2,
        wallets_per_cluster=5,
        num_legit=30,
        seed=7,
    )
    assert not tx.empty
    assert "chain" in tx.columns
    assert tx["chain"].nunique() >= 2
    assert set(labels["difficulty_tags"].astype(str).unique()) == {"L2,L6"}


def test_adversarial_benchmark_returns_expected_metric_keys() -> None:
    results = run_adversarial_detection_benchmark(
        detector_factory=lambda: SybilDetector(min_cluster_size=3, min_samples=2),
        levels=[1, 2],
        num_clusters=2,
        wallets_per_cluster=5,
        num_legit=30,
        seed=11,
    )
    assert set(results.keys()) == {"level_1", "level_2"}
    for metrics in results.values():
        assert set(metrics.keys()) == {
            "address_precision",
            "address_recall",
            "address_f1",
            "cluster_precision",
            "cluster_recall",
            "cluster_f1",
            "n_predicted_sybil_addresses",
        }
