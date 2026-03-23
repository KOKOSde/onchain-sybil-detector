"""Synthetic/adversarial dataset generators bundled with sybil_detector."""

from .adversarial_simulator_osd import (
    generate_adversarial_sybils,
    run_adversarial_detection_benchmark,
)
from .synthetic_generator import generate_synthetic_sybil_network

__all__ = [
    "generate_synthetic_sybil_network",
    "generate_adversarial_sybils",
    "run_adversarial_detection_benchmark",
]
