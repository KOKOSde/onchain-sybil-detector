"""onchain-sybil-detector package."""

from typing import Optional

from .benchmark import run_benchmark
from .clustering import SybilDetector
from .datasets.adversarial_simulator_osd import generate_adversarial_sybils
from .datasets.synthetic_generator import generate_synthetic_sybil_network as _generate_synthetic_sybil_network
from .feature_engineering import extract_features


def generate_synthetic_sybil_network(
    num_legit: int = 500,
    num_sybil_clusters: int = 10,
    addrs_per_cluster: int = 20,
    seed: int = 42,
    num_clusters: Optional[int] = None,
    wallets_per_cluster: Optional[int] = None,
):
    """Compatibility wrapper with alias-friendly parameter names."""

    if num_clusters is not None:
        num_sybil_clusters = int(num_clusters)
    if wallets_per_cluster is not None:
        addrs_per_cluster = int(wallets_per_cluster)
    return _generate_synthetic_sybil_network(
        num_legit=int(num_legit),
        num_sybil_clusters=int(num_sybil_clusters),
        addrs_per_cluster=int(addrs_per_cluster),
        seed=int(seed),
    )


__all__ = [
    "SybilDetector",
    "extract_features",
    "run_benchmark",
    "generate_synthetic_sybil_network",
    "generate_adversarial_sybils",
]
