# Datasets

This folder contains offline-safe datasets used by `onchain-sybil-detector`.

## Files

- `synthetic_generator.py`
  - Generates realistic synthetic transactions and labels for fully offline experiments.
  - Includes both legitimate users and coordinated Sybil clusters with noise.
- `known_sybil_addresses.csv`
  - Curated list of about 100 labeled addresses from public-report categories:
    - Hop Protocol Sybil investigations
    - Arbitrum public Sybil report categories
    - LayerZero public report categories

## Labeling Notes

- The CSV is a **self-contained educational starter set** and intentionally uses
  placeholder/public-example-style addresses to keep the repository redistributable.
- Column `source_type` identifies the report category.
- Column `source_reference` links to the source family where reports are publicly discussed.
- Column `is_placeholder` marks addresses that should be replaced by exact report addresses
  when you build a production-grade benchmark.

## Reproducibility

- Synthetic data generation is deterministic via `seed`.
- All tests and notebooks in this repository use synthetic data only (no network/API calls).
