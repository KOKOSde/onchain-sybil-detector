# onchain-sybil-detector: Open-Source Sybil Attack Detection for Blockchain Networks

![MIT](https://img.shields.io/badge/license-MIT-green)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

## Problem
Detecting multi-account abuse is critical for airdrops, DeFi protocols, and exchanges. Enterprise tools cost $100k+/year. This is the free alternative.

## How It Works
`onchain-sybil-detector` uses an offline-first behavioral pipeline:
1. Build address-level behavior features from transactions.
2. Run HDBSCAN to discover coordinated groups.
3. Score coordination using temporal, gas, funding, and sequencing evidence.
4. Return explainable Sybil probabilities per address and cluster.

## Architecture
```text
+-------------------+       +----------------------+       +----------------------+
| Data Ingestion    | ----> | Feature Engineering  | ----> | HDBSCAN + Refinement |
| - RPC/Etherscan   |       | - 25 behavior feats  |       | - cluster_id         |
| - SQLite cache    |       | - hour hist vectors  |       | - sybil_probability  |
| - Offline fallback|       | - funding signatures |       | - evidence breakdown |
+-------------------+       +----------------------+       +----------+-----------+
                                                                        |
                                                                        v
                                                           +--------------------------+
                                                           | Reports + Visualization  |
                                                           | - pyvis graph HTML       |
                                                           | - cluster explanations    |
                                                           | - benchmark vs baselines |
                                                           +--------------------------+
```

## Quickstart
```python
from datasets.synthetic_generator import generate_synthetic_sybil_network
from sybil_detector import SybilDetector, extract_features

tx, labels = generate_synthetic_sybil_network(seed=42)
clusters = SybilDetector().fit_predict(extract_features(tx))
```

## Setup: API Keys (Optional)
Works out of the box with synthetic data - no keys needed.

For real blockchain data, get free keys:
- Etherscan: https://etherscan.io/myapikey (works for ETH + BNB + more chains)
- Alchemy: https://dashboard.alchemy.com/signup (free tier: 300M compute units/month)

Create and populate environment variables:
```bash
cp .env.example .env
# edit .env and insert your keys
export ETHERSCAN_API_KEY=your_key
```

## Install
```bash
python3.11 -m pip install -e .
```

## Results (Synthetic, Offline)
| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| onchain-sybil-detector (address-level) | 1.00 | 0.83 | 0.91 |
| Same funder baseline | 0.10 | 0.50 | 0.17 |
| Same gas baseline | 0.78 | 0.74 | 0.76 |
| Same timing baseline | 0.65 | 0.65 | 0.65 |

## Who This Is For
- Crypto exchanges (for example Coinbase risk teams)
- DeFi protocols designing anti-Sybil airdrops
- Airdrop designers and token distribution analysts
- Security researchers and investigators

## Screenshots
- `notebooks/analysis.ipynb` produces:
  - Interactive network graph (`pyvis`) with flagged clusters
  - Cluster coordination report summary (`HTML`)

## Repository Layout
```text
onchain-sybil-detector/
├── pyproject.toml
├── README.md
├── LICENSE
├── Makefile
├── .env.example
├── .github/workflows/ci.yml
├── src/sybil_detector/
├── datasets/
├── tests/
├── notebooks/
└── examples/
```
