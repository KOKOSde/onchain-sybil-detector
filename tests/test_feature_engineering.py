import numpy as np

from sybil_detector.datasets.synthetic_generator import generate_synthetic_sybil_network
from sybil_detector.feature_engineering import FEATURE_COLUMNS, extract_features


def test_extract_features_has_all_required_columns_and_no_nan() -> None:
    tx, _ = generate_synthetic_sybil_network(
        num_legit=180,
        num_sybil_clusters=6,
        addrs_per_cluster=14,
        seed=7,
    )
    features = extract_features(tx)

    expected = {"address", *FEATURE_COLUMNS}
    assert expected.issubset(set(features.columns))
    assert features.shape[0] == tx["address"].nunique()

    numeric = features.drop(columns=["address", "common_funder_address", "hour_histogram"])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
