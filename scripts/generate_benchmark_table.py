#!/usr/bin/env python3
"""Generate adversarial benchmark artifacts and markdown table from code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from sybil_detector.datasets.adversarial_simulator_osd import generate_adversarial_sybils
from sybil_detector import SybilDetector, extract_features


LEVEL_DESCRIPTIONS: Dict[int, str] = {
    1: "Naive (same funder/gas/timing)",
    2: "Randomized timing",
    3: "Indirect funding (2-hop)",
    4: "Mixed gas behavior",
    5: "Intentional cluster splitting",
    6: "Chain hopping",
    7: "Burner wallets",
    8: "Delayed coordination",
}


def _compute_level_metrics(level: int, seed: int, num_clusters: int, wallets_per_cluster: int) -> Tuple[float, float, float]:
    tx_df, labels_df = generate_adversarial_sybils(
        difficulty=level,
        num_clusters=num_clusters,
        wallets_per_cluster=wallets_per_cluster,
        seed=seed,
    )
    features = extract_features(tx_df)
    preds_df = SybilDetector().fit_predict(features)

    merged = preds_df.merge(labels_df[["address", "is_sybil"]], on="address", how="inner")
    y_true = merged["is_sybil"].astype(int)
    y_pred = (merged["sybil_probability"] >= 0.5).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return precision, recall, f1


def generate_rows(seed: int, num_clusters: int, wallets_per_cluster: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for level in range(1, 9):
        p, r, f1 = _compute_level_metrics(
            level=level,
            seed=seed,
            num_clusters=num_clusters,
            wallets_per_cluster=wallets_per_cluster,
        )
        rows.append(
            {
                "level": level,
                "description": LEVEL_DESCRIPTIONS[level],
                "precision": p,
                "recall": r,
                "f1": f1,
                "detection_rate": r * 100.0,
            }
        )
    return rows


def to_markdown(rows: List[Dict[str, float]]) -> str:
    lines = [
        "| Difficulty | Description | Precision | Recall | F1 | Detection Rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| Level {level} | {description} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {detection_rate:.2f}% |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate adversarial benchmark table artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--wallets-per-cluster", type=int, default=10)
    parser.add_argument("--out-json", default="demo_outputs_osd/adversarial_benchmark.json")
    parser.add_argument("--out-csv", default="demo_outputs_osd/adversarial_benchmark.csv")
    parser.add_argument("--out-md", default="demo_outputs_osd/adversarial_benchmark_table.md")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()

    rows = generate_rows(
        seed=int(args.seed),
        num_clusters=int(args.num_clusters),
        wallets_per_cluster=int(args.wallets_per_cluster),
    )

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    level_payload = {
        "seed": int(args.seed),
        "num_clusters": int(args.num_clusters),
        "wallets_per_cluster": int(args.wallets_per_cluster),
        "levels": {
            f"level_{row['level']}": {
                "address_precision": row["precision"],
                "address_recall": row["recall"],
                "address_f1": row["f1"],
                "detection_rate": row["detection_rate"],
                "description": row["description"],
            }
            for row in rows
        },
    }
    out_json.write_text(json.dumps(level_payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    md_table = to_markdown(rows)
    out_md.write_text(md_table + "\n", encoding="utf-8")

    if args.update_readme:
        readme_path = Path("README.md")
        if readme_path.exists():
            content = readme_path.read_text(encoding="utf-8")
            start = "<!-- ADVERSARIAL_TABLE_START -->"
            end = "<!-- ADVERSARIAL_TABLE_END -->"
            if start in content and end in content:
                head, tail = content.split(start, 1)
                _, rest = tail.split(end, 1)
                content = head + start + "\n" + md_table + "\n" + end + rest
                readme_path.write_text(content, encoding="utf-8")

    print(md_table)
    print("")
    print("Wrote:", out_json)
    print("Wrote:", out_csv)
    print("Wrote:", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
