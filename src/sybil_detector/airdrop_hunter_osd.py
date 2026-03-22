"""Airdrop hunter mode for campaign/referral/faucet Sybil abuse analysis."""

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .chains_osd import attach_chain_defaults, detect_cross_chain_coordination, normalize_chain_name
from .clustering import SybilDetector
from .feature_engineering import extract_features


def _normalize_addresses(addresses: Iterable[str]) -> List[str]:
    return [str(a).strip().lower() for a in addresses if str(a).strip()]


def _filter_participant_transactions(transactions: pd.DataFrame, addresses: List[str]) -> pd.DataFrame:
    if transactions.empty:
        return transactions.copy()

    tx = transactions.copy()
    for col in ["address", "from_addr", "to_addr"]:
        if col in tx.columns:
            tx[col] = tx[col].astype(str).str.lower()

    mask = pd.Series(False, index=tx.index)
    for col in ["address", "from_addr", "to_addr"]:
        if col in tx.columns:
            mask = mask | tx[col].isin(addresses)
    return tx.loc[mask].copy().reset_index(drop=True)


def _wallet_evidence_row(row: pd.Series) -> Dict[str, float]:
    gas_cv = float(row.get("gas_price_cv", 0.0))
    burst_ratio = float(row.get("burst_ratio", 0.0))
    top_source = float(row.get("pct_funds_from_top_source", 0.0))
    timing_entropy = float(row.get("hour_of_day_entropy", 0.0))

    return {
        "funding_concentration": float(np.clip(top_source, 0.0, 1.0)),
        "burst_activity": float(np.clip(burst_ratio, 0.0, 1.0)),
        "gas_consistency": float(np.clip(1.0 - gas_cv, 0.0, 1.0)),
        "timing_regularization": float(np.clip(1.0 - timing_entropy / np.log2(24.0), 0.0, 1.0)),
    }


def run_airdrop_hunter(
    participant_addresses: Iterable[str],
    transactions: pd.DataFrame,
    chain: str = "ethereum",
    min_cluster_size: int = 3,
    min_samples: int = 2,
    confidence_threshold: float = 0.6,
) -> Dict[str, object]:
    """Detect likely airdrop farmer clusters and wallet-level suspicion.

    Input transactions are expected to follow core transaction schema and can include optional
    chain-specific columns. The function runs fully offline.
    """

    chain_name = normalize_chain_name(chain)
    participants = _normalize_addresses(participant_addresses)
    if not participants:
        return {
            "chain": chain_name,
            "participant_count": 0,
            "likely_farmer_clusters": [],
            "wallet_scores": [],
            "wallet_evidence": {},
            "campaign_summary": {
                "estimated_sybil_participants": 0,
                "estimated_sybil_percentage": 0.0,
                "confidence_threshold": confidence_threshold,
            },
            "marginal_members": [],
            "cross_chain_signals": {},
        }

    tx = _filter_participant_transactions(transactions, participants)
    if "chain" not in tx.columns:
        tx = attach_chain_defaults(tx, chain_name)
    else:
        tx["chain"] = tx["chain"].astype(str).str.lower().replace({"eth": "ethereum"})

    features = extract_features(tx)
    features = features[features["address"].isin(participants)].reset_index(drop=True)

    detector = SybilDetector(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusters = detector.fit_predict(features)

    merged = features.merge(
        clusters[["address", "cluster_id", "sybil_probability", "coordination_breakdown"]],
        on="address",
        how="left",
    )
    merged["cluster_id"] = merged["cluster_id"].fillna(-1).astype(int)
    merged["sybil_probability"] = merged["sybil_probability"].fillna(0.0).astype(float)

    wallet_evidence: Dict[str, Dict[str, float]] = {}
    suspicion_scores = []
    for _, row in merged.iterrows():
        addr = str(row["address"])
        evidence = _wallet_evidence_row(row)
        wallet_evidence[addr] = evidence
        score = 0.45 * float(row["sybil_probability"])
        score += 0.25 * evidence["funding_concentration"]
        score += 0.15 * evidence["gas_consistency"]
        score += 0.15 * evidence["burst_activity"]
        suspicion_scores.append(float(np.clip(score, 0.0, 1.0)))

    merged["wallet_suspicion_score"] = suspicion_scores
    merged = merged.sort_values("wallet_suspicion_score", ascending=False).reset_index(drop=True)

    # Likely farmer clusters (cluster-level confidence from wallet scores).
    cluster_rows = []
    for cluster_id, grp in merged[merged["cluster_id"] != -1].groupby("cluster_id"):
        avg_score = float(grp["wallet_suspicion_score"].mean())
        if avg_score < confidence_threshold:
            continue
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "wallet_count": int(len(grp)),
                "avg_wallet_suspicion": avg_score,
                "max_wallet_suspicion": float(grp["wallet_suspicion_score"].max()),
                "addresses": grp["address"].tolist(),
            }
        )

    cluster_rows = sorted(cluster_rows, key=lambda x: x["avg_wallet_suspicion"], reverse=True)

    flagged_wallets = merged[merged["wallet_suspicion_score"] >= confidence_threshold]
    marginal = merged[
        (merged["wallet_suspicion_score"] >= confidence_threshold * 0.6)
        & (merged["wallet_suspicion_score"] < confidence_threshold)
    ]

    cross_chain = detect_cross_chain_coordination(tx) if "chain" in tx.columns else {}

    result = {
        "chain": chain_name,
        "participant_count": int(len(participants)),
        "likely_farmer_clusters": cluster_rows,
        "wallet_scores": merged[["address", "cluster_id", "wallet_suspicion_score", "sybil_probability"]]
        .to_dict(orient="records"),
        "wallet_evidence": wallet_evidence,
        "campaign_summary": {
            "estimated_sybil_participants": int(len(flagged_wallets)),
            "estimated_sybil_percentage": float(len(flagged_wallets) / max(len(participants), 1)),
            "confidence_threshold": float(confidence_threshold),
            "cluster_count": int(len(cluster_rows)),
        },
        "marginal_members": marginal[["address", "wallet_suspicion_score", "cluster_id"]].to_dict(orient="records"),
        "cross_chain_signals": cross_chain,
    }
    return result


def detect_campaign_participants(transactions: pd.DataFrame, campaign_contract: Optional[str]) -> List[str]:
    """Infer participant addresses for a campaign contract from transaction records."""

    if transactions.empty:
        return []

    tx = transactions.copy()
    for col in ["address", "from_addr", "to_addr"]:
        if col in tx.columns:
            tx[col] = tx[col].astype(str).str.lower()

    if campaign_contract:
        target = str(campaign_contract).lower()
        mask = pd.Series(False, index=tx.index)
        if "to_addr" in tx.columns:
            mask = mask | (tx["to_addr"] == target)
        if "from_addr" in tx.columns:
            mask = mask | (tx["from_addr"] == target)
        filtered = tx.loc[mask]
    else:
        filtered = tx

    candidates = set()
    for col in ["address", "from_addr", "to_addr"]:
        if col in filtered.columns:
            candidates.update(filtered[col].dropna().astype(str).str.lower().tolist())

    if campaign_contract:
        candidates.discard(str(campaign_contract).lower())

    return sorted([x for x in candidates if x.startswith("0x")])


def scan_airdrop_campaign(
    transactions: pd.DataFrame,
    campaign_contract: Optional[str],
    chain: str = "ethereum",
    participant_addresses: Optional[Iterable[str]] = None,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    confidence_threshold: float = 0.6,
) -> Dict[str, object]:
    """Campaign-first wrapper for airdrop abuse scans."""

    participants = (
        _normalize_addresses(participant_addresses)
        if participant_addresses is not None
        else detect_campaign_participants(transactions, campaign_contract)
    )
    result = run_airdrop_hunter(
        participant_addresses=participants,
        transactions=transactions,
        chain=chain,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        confidence_threshold=confidence_threshold,
    )
    result["campaign_contract"] = str(campaign_contract).lower() if campaign_contract else None
    return result


__all__ = ["run_airdrop_hunter", "detect_campaign_participants", "scan_airdrop_campaign"]
