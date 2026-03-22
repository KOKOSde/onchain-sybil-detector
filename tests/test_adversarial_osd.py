import numpy as np
import pytest

from datasets.adversarial_simulator_osd import (
    generate_adversarial_sybils,
    run_adversarial_detection_benchmark,
)
from sybil_detector import SybilDetector
from sybil_detector.benchmark import run_benchmark

REQUIRED_TX_COLS = {
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
REQUIRED_LABEL_COLS = {"address", "is_sybil", "cluster_id", "operator_id", "difficulty_level", "difficulty_tags"}


@pytest.mark.parametrize("level", list(range(1, 9)))
def test_each_difficulty_level_has_expected_schema(level: int) -> None:
    tx, labels = generate_adversarial_sybils(
        difficulty=level,
        num_clusters=3,
        wallets_per_cluster=6,
        num_legit=60,
        seed=42,
    )
    assert not tx.empty
    assert not labels.empty
    assert REQUIRED_TX_COLS.issubset(set(tx.columns))
    assert REQUIRED_LABEL_COLS.issubset(set(labels.columns))
    assert int(labels["difficulty_level"].max()) == level


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5, 6, 7, 8])
def test_each_difficulty_level_contains_sybil_labels(level: int) -> None:
    _, labels = generate_adversarial_sybils(
        difficulty=level,
        num_clusters=2,
        wallets_per_cluster=5,
        num_legit=20,
        seed=21,
    )
    sybil_count = int((labels["is_sybil"] == 1).sum())
    assert sybil_count > 0


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


def _address_recall(level: int) -> float:
    tx, labels = generate_adversarial_sybils(
        difficulty=level,
        num_clusters=4,
        wallets_per_cluster=8,
        num_legit=120,
        seed=42,
    )
    metrics = run_benchmark(SybilDetector(min_cluster_size=3, min_samples=2), tx, labels)
    return float(metrics["address"]["recall"])


def test_higher_difficulty_generally_reduces_detection_rates() -> None:
    easy_recall = float(np.mean([_address_recall(1), _address_recall(2)]))
    hard_recall = float(np.mean([_address_recall(3), _address_recall(8)]))
    assert hard_recall < easy_recall


def test_level_8_has_larger_sybil_activity_span_than_level_1() -> None:
    tx1, labels1 = generate_adversarial_sybils(
        difficulty=1,
        num_clusters=3,
        wallets_per_cluster=6,
        num_legit=30,
        seed=99,
    )
    tx8, labels8 = generate_adversarial_sybils(
        difficulty=8,
        num_clusters=3,
        wallets_per_cluster=6,
        num_legit=30,
        seed=99,
    )

    sybil1 = set(labels1.loc[labels1["is_sybil"] == 1, "address"].astype(str).str.lower())
    sybil8 = set(labels8.loc[labels8["is_sybil"] == 1, "address"].astype(str).str.lower())

    span1 = int(tx1[tx1["address"].isin(sybil1)]["timestamp"].max()) - int(
        tx1[tx1["address"].isin(sybil1)]["timestamp"].min()
    )
    span8 = int(tx8[tx8["address"].isin(sybil8)]["timestamp"].max()) - int(
        tx8[tx8["address"].isin(sybil8)]["timestamp"].min()
    )

    assert span8 > span1
