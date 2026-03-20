"""Benchmarking utilities for Sybil detection quality assessment."""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .feature_engineering import extract_features


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true.astype(int),
        y_pred.astype(int),
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _cluster_level_metrics(eval_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    true_clusters = set(
        eval_df.loc[(eval_df["is_sybil"] == 1) & (eval_df["cluster_id_true"] != -1), "cluster_id_true"]
        .astype(int)
        .tolist()
    )

    cluster_summary = (
        eval_df[eval_df["cluster_id_pred"] != -1]
        .groupby("cluster_id_pred", as_index=False)
        .agg(
            size=("address", "count"),
            mean_sybil_probability=("sybil_probability", "mean"),
            sybil_ratio=("is_sybil", "mean"),
            predominant_true_cluster=("cluster_id_true", lambda s: int(pd.Series(s).mode().iloc[0])),
        )
    )

    pred_clusters = set(
        cluster_summary.loc[
            cluster_summary["mean_sybil_probability"] >= threshold,
            "cluster_id_pred",
        ]
        .astype(int)
        .tolist()
    )

    if not pred_clusters:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = 0
    matched_true = set()
    for cid in sorted(pred_clusters):
        row = cluster_summary.loc[cluster_summary["cluster_id_pred"] == cid].iloc[0]
        if float(row["sybil_ratio"]) >= 0.5:
            tp += 1
            true_cid = int(row["predominant_true_cluster"])
            if true_cid in true_clusters:
                matched_true.add(true_cid)

    precision = float(tp / len(pred_clusters)) if pred_clusters else 0.0
    recall = float(len(matched_true) / len(true_clusters)) if true_clusters else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _compute_baselines(features: pd.DataFrame, y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
    common_funder = features.get("common_funder_address", pd.Series([""] * len(features))).astype(str)
    pct_top = pd.to_numeric(features.get("pct_funds_from_top_source", 0.0), errors="coerce").fillna(0.0)
    funding_sources = pd.to_numeric(features.get("funding_source_count", 0.0), errors="coerce").fillna(0.0)
    gas_cv = pd.to_numeric(features.get("gas_price_cv", 0.0), errors="coerce").fillna(0.0)
    gas_mode = pd.to_numeric(features.get("gas_price_mode_ratio", 0.0), errors="coerce").fillna(0.0)
    burst = pd.to_numeric(features.get("burst_ratio", 0.0), errors="coerce").fillna(0.0)
    day_entropy = pd.to_numeric(features.get("day_of_week_entropy", 0.0), errors="coerce").fillna(0.0)

    same_funder_pred = (
        (pct_top.to_numpy(dtype=float) >= 0.55)
        & (funding_sources.to_numpy(dtype=float) <= 5.0)
        & (~common_funder.isin(["", "none", "nan"]).to_numpy())
    )
    same_gas_pred = (
        (gas_cv.to_numpy(dtype=float) <= 0.08)
        & (gas_mode.to_numpy(dtype=float) <= 0.25)
    )
    same_timing_pred = (
        (burst.to_numpy(dtype=float) >= 0.10)
        & (day_entropy.to_numpy(dtype=float) >= 1.8)
    )

    return {
        "same_funder": _binary_metrics(y_true, same_funder_pred.astype(int)),
        "same_gas": _binary_metrics(y_true, same_gas_pred.astype(int)),
        "same_timing": _binary_metrics(y_true, same_timing_pred.astype(int)),
    }


def run_benchmark(detector, transactions: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, object]:
    """Run address/cluster benchmark and compare against naive baselines."""

    features = extract_features(transactions)
    preds = detector.fit_predict(features)

    eval_df = features[["address"]].merge(
        labels.rename(columns={"cluster_id": "cluster_id_true"}),
        on="address",
        how="left",
    )
    eval_df["is_sybil"] = eval_df["is_sybil"].fillna(0).astype(int)
    eval_df["cluster_id_true"] = eval_df["cluster_id_true"].fillna(-1).astype(int)

    eval_df = eval_df.merge(
        preds.rename(columns={"cluster_id": "cluster_id_pred"}),
        on="address",
        how="left",
    )
    eval_df["cluster_id_pred"] = eval_df["cluster_id_pred"].fillna(-1).astype(int)
    eval_df["sybil_probability"] = eval_df["sybil_probability"].fillna(0.0).astype(float)

    threshold = 0.70
    y_true = eval_df["is_sybil"].to_numpy(dtype=int)
    y_pred = (
        (eval_df["cluster_id_pred"].to_numpy(dtype=int) != -1)
        & (eval_df["sybil_probability"].to_numpy(dtype=float) >= threshold)
    ).astype(int)

    address_metrics = _binary_metrics(y_true, y_pred)
    cluster_metrics = _cluster_level_metrics(eval_df, threshold=threshold)
    baselines = _compute_baselines(features, y_true)

    n_predicted_clusters = int(
        eval_df.loc[
            (eval_df["cluster_id_pred"] != -1) & (eval_df["sybil_probability"] >= threshold),
            "cluster_id_pred",
        ]
        .nunique()
    )

    return {
        "address": address_metrics,
        "cluster": cluster_metrics,
        "baselines": baselines,
        "decision_threshold": float(threshold),
        "n_addresses": int(len(eval_df)),
        "n_predicted_sybil_addresses": int(y_pred.sum()),
        "n_predicted_clusters": n_predicted_clusters,
    }


__all__ = ["run_benchmark"]
