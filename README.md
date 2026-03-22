# onchain-sybil-detector
Detect coordinated multi-wallet abuse on any EVM chain. Evidence-driven. Explainable. Open source.

![MIT](https://img.shields.io/badge/license-MIT-green)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests passing](https://img.shields.io/badge/tests-13%20passing-brightgreen)
![Supported chains](https://img.shields.io/badge/chains-ETH%20%7C%20Base%20%7C%20BNB%20%7C%20Arbitrum%20%7C%20Optimism%20%7C%20Polygon-1f6feb)

## Why This Exists
Detecting multi-account abuse is critical for airdrops, DeFi protocols, and exchanges. Enterprise Sybil investigation platforms can cost $100k+/year. `onchain-sybil-detector` is a free, open-source alternative focused on behavioral clustering and evidence-driven investigation.

## Architecture
```text
+-------------------+    +----------------------+    +----------------------+    +----------------------+    +----------------------+
| Data Ingestion    | -> | Feature Engineering  | -> | HDBSCAN Clustering   | -> | Evidence Scoring     | -> | Analyst Report       |
| (RPC/API/cache)   |    | (25 behavior feats)  |    | (+ label refinement) |    | (timing/gas/funding) |    | (HTML/JSON/Markdown) |
+-------------------+    +----------------------+    +----------------------+    +----------------------+    +----------------------+
```

## Quickstart
```bash
python -m pip install -e .
```

```python
from datasets.synthetic_generator import generate_synthetic_sybil_network
from sybil_detector import SybilDetector, extract_features

tx, labels = generate_synthetic_sybil_network(seed=42)
clusters = SybilDetector().fit_predict(extract_features(tx))
print(clusters.head())
```

## Setup: API Keys (Optional)
Works out of the box with synthetic data, no keys required.

For live chain ingestion, get free keys:
- Etherscan-family API key: https://etherscan.io/myapikey (works for ETH + BNB + other Etherscan-family explorers)
- Alchemy: https://dashboard.alchemy.com/signup

```bash
cp .env.example .env
# edit .env with your keys
export ETHERSCAN_API_KEY=your_key
export ALCHEMY_API_KEY=your_key
```

## CLI Quickstart
```bash
python -m sybil_detector.cli_osd simulate --difficulty 1 --num-clusters 5 --wallets-per-cluster 10 --out /tmp/osd_synthetic.csv
python -m sybil_detector.cli_osd scan --addresses /tmp/osd_synthetic.csv --chain eth --out /tmp/osd_report.html
python -m sybil_detector.cli_osd explain --wallets 0xAAA,0xBBB,0xCCC --chain eth --transactions /tmp/osd_synthetic.csv
```

## Airdrop Hunter Mode
Specialized detection mode for:
- airdrop campaigns
- quest/task farming
- faucet abuse
- referral program abuse
- reward/incentive farming

Example:
```bash
python -m sybil_detector.cli_osd airdrop-scan --contract 0x123... --chain base --out /tmp/airdrop_report.html
```

Output includes likely farmer clusters, per-wallet suspicion score, marginal members, and campaign-level estimated abuse.

## Multi-Chain Support
| Chain | Alias | Explorer API Base | Typical block time |
|---|---|---|---:|
| Ethereum | `eth` | `https://api.etherscan.io/api` | 12.0s |
| Base | `base` | `https://api.basescan.org/api` | 2.0s |
| BNB Chain | `bnb`, `bsc` | `https://api.bscscan.com/api` | 3.0s |
| Arbitrum | `arb` | `https://api.arbiscan.io/api` | 0.26s |
| Optimism | `op` | `https://api-optimistic.etherscan.io/api` | 2.0s |
| Polygon | `matic` | `https://api.polygonscan.com/api` | 2.1s |

Cross-chain coordination signals include bridge timing correlation, mirrored gas behavior, synchronized windows, repeated wallet generation patterns, and common funding sources.

## Adversarial Benchmark Results
Real output from `demo_outputs_osd/osd_benchmark.csv`:

| Difficulty | Address Precision | Address Recall | Address F1 | Cluster F1 |
|---|---:|---:|---:|---:|
| Level 1 | 1.000 | 0.875 | 0.933 | 1.000 |
| Level 2 | 1.000 | 0.717 | 0.835 | 0.947 |
| Level 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| Level 4 | 1.000 | 0.833 | 0.909 | 0.947 |
| Level 5 | 1.000 | 0.875 | 0.933 | 0.400 |
| Level 6 | 1.000 | 0.875 | 0.933 | 1.000 |
| Level 7 | 1.000 | 0.750 | 0.857 | 1.000 |
| Level 8 | 0.000 | 0.000 | 0.000 | 0.000 |

Raw benchmark transcript (`demo_outputs_osd/adversarial_benchmark_osd.txt`):
```text
Level 1: precision=1.000, recall=0.942, f1=0.970
Level 2: precision=1.000, recall=0.800, f1=0.889
Level 3: precision=0.000, recall=0.000, f1=0.000
...
```

## Why These Wallets Are Linked
Real CLI output (`demo_outputs_osd/cli_explain_osd.txt`):
```json
{
  "evidence_sentences": [
    "Transact in same minute-buckets across 95 windows.",
    "Near-identical gas strategy (median gas_price CV=0.062, gas_used CV=0.044).",
    "Recurring contract interaction sequence similarity: 1.00."
  ],
  "confidence_score": 0.6200000000000001,
  "linkage_strength": "moderate"
}
```

## Analyst Report
Generated artifacts (real run):
- `demo_outputs_osd/reports/osd_demo_analyst_report.html`
- `demo_outputs_osd/reports/osd_demo_analyst_report.json`
- `demo_outputs_osd/reports/osd_demo_graph.html`

Sanitized report snippet:
```text
Cluster 3: 10 wallets funded by 0x...f00003, confidence=0.87, gas CV=0.080.
```

## Screenshots
- Interactive network graph HTML: `demo_outputs_osd/reports/osd_demo_graph.html`
- Cluster report HTML: `demo_outputs_osd/reports/osd_demo_analyst_report.html`

## Demo Output
Real command outputs:
```text
{"transactions": "/tmp/osd_synthetic.csv", "labels": "/tmp/osd_synthetic_labels.csv", "rows": 14536}
{"output": "/tmp/osd_report.html", "clusters": 50}
{"output": "/tmp/osd_benchmark.csv", "levels": 8}
```

Real full-suite test output (`demo_outputs_osd/test_output_osd_final.txt`):
```text
collected 13 items
...
======================== 13 passed in 67.09s (0:01:07) =========================
```

## Who This Is For
- Crypto exchanges (for example Coinbase risk and trust teams)
- DeFi protocols
- Airdrop designers
- Security researchers
- DAO governance and anti-abuse teams

## Results Table (Synthetic)
From `demo_outputs_osd/synthetic_benchmark_osd.json`:

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| OSD detector (address-level) | 1.000 | 0.830 | 0.907 |
| Baseline: same funder | 0.100 | 0.500 | 0.167 |
| Baseline: same gas | 0.780 | 0.745 | 0.762 |
| Baseline: same timing | 0.655 | 0.645 | 0.650 |

## Related Projects
- identity-risk-engine: https://github.com/KOKOSde/identity-risk-engine
- LocalMod: https://github.com/KOKOSde/LocalMod

## Contributing
Issues and PRs are welcome. Prefer small, test-backed changes and include offline reproducibility for new detection logic.

## License
MIT
