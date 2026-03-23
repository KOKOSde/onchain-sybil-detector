"""Plain-English wallet linkage explainer."""

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .feature_engineering import extract_features


def _normalize_wallets(wallets: Iterable[str]) -> List[str]:
    values = [str(w).strip().lower() for w in wallets if str(w).strip()]
    unique = []
    seen = set()
    for w in values:
        if w in seen:
            continue
        seen.add(w)
        unique.append(w)
    return unique


def _sequence_similarity(sequences: List[List[str]]) -> float:
    if len(sequences) < 2:
        return 0.0

    sims = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            a = sequences[i]
            b = sequences[j]
            if not a and not b:
                sims.append(0.0)
                continue
            if not a or not b:
                sims.append(0.0)
                continue
            set_a = set(a)
            set_b = set(b)
            inter = len(set_a & set_b)
            union = len(set_a | set_b) or 1
            sims.append(float(inter / union))
    return float(np.mean(sims)) if sims else 0.0


def explain_wallet_linkage(
    wallets: Iterable[str],
    transactions: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    chains_data: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Generate linkage evidence sentences for 2+ wallets.

    Returns evidence sentences, confidence score, and linkage strength.
    """

    normalized = _normalize_wallets(wallets)

    def insufficient_history_result() -> Dict[str, object]:
        sentence = "Insufficient transaction history for analysis."
        return {
            "wallets": normalized,
            "evidence": [{"type": "insufficient_history", "sentence": sentence, "score": 0.0}],
            "evidence_sentences": [sentence],
            "confidence_score": 0.0,
            "linkage_strength": "none",
        }

    if len(normalized) < 2:
        return {
            "wallets": normalized,
            "evidence_sentences": ["Need at least two wallets for linkage analysis."],
            "confidence_score": 0.0,
            "linkage_strength": "none",
        }

    tx = transactions.copy()
    for col in ["address", "from_addr", "to_addr", "input_data_prefix"]:
        if col in tx.columns:
            tx[col] = tx[col].astype(str).str.lower()
    if "from_addr" not in tx.columns:
        tx["from_addr"] = ""
    if "to_addr" not in tx.columns:
        tx["to_addr"] = ""
    if "input_data_prefix" not in tx.columns:
        tx["input_data_prefix"] = "0x"
    if "address" not in tx.columns:
        # Default actor identity to sender when address-level annotation is absent.
        tx["address"] = tx["from_addr"].astype(str).str.lower()
    tx["timestamp"] = pd.to_numeric(tx.get("timestamp", 0), errors="coerce").fillna(0).astype(np.int64)
    tx["gas_price"] = pd.to_numeric(tx.get("gas_price", 0), errors="coerce").fillna(0.0)

    subset = tx[
        tx["address"].isin(normalized)
        | tx["from_addr"].isin(normalized)
        | tx["to_addr"].isin(normalized)
    ].copy()

    if subset.empty:
        return insufficient_history_result()

    for wallet in normalized:
        wallet_history = subset[
            (subset["address"] == wallet) | (subset["from_addr"] == wallet) | (subset["to_addr"] == wallet)
        ]
        if wallet_history.empty:
            return insufficient_history_result()

    if features is None:
        try:
            fdf = extract_features(subset)
        except Exception:
            fdf = pd.DataFrame()
    else:
        fdf = features.copy()

    if not fdf.empty and "address" in fdf.columns:
        fdf["address"] = fdf["address"].astype(str).str.lower()
        fdf = fdf[fdf["address"].isin(normalized)].copy()
    elif features is not None:
        return insufficient_history_result()

    if features is not None:
        if fdf.empty or "address" not in fdf.columns:
            return insufficient_history_result()
        for wallet in normalized:
            row = fdf[fdf["address"] == wallet]
            if row.empty:
                return insufficient_history_result()
            numeric_cols = [c for c in row.columns if c not in {"address", "common_funder_address", "hour_histogram"}]
            if numeric_cols:
                non_zero_mass = float(
                    row[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs().sum(axis=1).iloc[0]
                )
                if non_zero_mass <= 0:
                    return insufficient_history_result()

    evidence: List[str] = []
    scores = []

    # 1) Same upstream funder within 8 minutes.
    incoming = subset[subset["to_addr"].isin(normalized) & (~subset["from_addr"].isin(normalized))]
    if not incoming.empty:
        common_funders = incoming.groupby("from_addr")["to_addr"].nunique().sort_values(ascending=False)
        if len(common_funders) > 0 and int(common_funders.iloc[0]) >= 2:
            funder = str(common_funders.index[0])
            grp = incoming[incoming["from_addr"] == funder]
            dt_sec = float(grp["timestamp"].max() - grp["timestamp"].min())
            if dt_sec <= 8 * 60:
                evidence.append(
                    "Funded from same upstream wallet ({}) within {:.1f} minutes.".format(funder, dt_sec / 60.0)
                )
                scores.append(0.28)

    # 2) Same minute-buckets across days.
    if not subset.empty:
        minute_bucket = pd.to_datetime(subset["timestamp"], unit="s", utc=True).dt.floor("min")
        subset = subset.assign(minute_bucket=minute_bucket)
        minute_overlap = subset.groupby("minute_bucket")["address"].nunique()
        shared_minutes = int((minute_overlap >= 2).sum())
        if shared_minutes >= 4:
            evidence.append("Transact in same minute-buckets across {} windows.".format(shared_minutes))
            scores.append(0.2)

    # 3) Near-identical gas strategy.
    if not fdf.empty and {"median_gas_price", "median_gas_used"}.issubset(fdf.columns):
        med_gp = pd.to_numeric(fdf["median_gas_price"], errors="coerce").fillna(0.0)
        med_gu = pd.to_numeric(fdf["median_gas_used"], errors="coerce").fillna(0.0)
        gp_cv = float(med_gp.std(ddof=0) / med_gp.mean()) if med_gp.mean() > 0 else 1.0
        gu_cv = float(med_gu.std(ddof=0) / med_gu.mean()) if med_gu.mean() > 0 else 1.0
        if gp_cv <= 0.15 and gu_cv <= 0.2:
            evidence.append(
                "Near-identical gas strategy (median gas_price CV={:.3f}, gas_used CV={:.3f}).".format(gp_cv, gu_cv)
            )
            scores.append(0.22)

    # 4) Contract interaction sequence similarity.
    seqs = []
    for wallet in normalized:
        wseq = (
            subset[subset["address"] == wallet]
            .sort_values("timestamp")
            .get("input_data_prefix", pd.Series([], dtype=str))
            .astype(str)
            .tolist()
        )
        seqs.append(wseq)
    seq_sim = _sequence_similarity(seqs)
    if seq_sim >= 0.55:
        evidence.append("Recurring contract interaction sequence similarity: {:.2f}.".format(seq_sim))
        scores.append(0.2)

    # 5) Bridge timing (cross-chain) correlation.
    if chains_data is not None and not chains_data.empty and "chain" in chains_data.columns:
        cdf = chains_data.copy()
        cdf["address"] = cdf["address"].astype(str).str.lower()
        cdf = cdf[cdf["address"].isin(normalized)]
        if not cdf.empty:
            first_seen = cdf.groupby(["address", "chain"]) ["timestamp"].min().reset_index()
            deltas = []
            for _, grp in first_seen.groupby("address"):
                vals = sorted(grp["timestamp"].tolist())
                if len(vals) >= 2:
                    deltas.extend(np.diff(vals))
            if deltas and float(np.mean(np.array(deltas) <= 180)) >= 0.5:
                evidence.append("Synchronized bridge timing observed across chains (<=3 minute deltas).")
                scores.append(0.1)

    confidence = float(np.clip(sum(scores), 0.0, 1.0))
    if confidence >= 0.75:
        linkage = "strong"
    elif confidence >= 0.45:
        linkage = "moderate"
    elif confidence > 0:
        linkage = "weak"
    else:
        linkage = "none"

    if not evidence:
        evidence = ["No strong direct linkage evidence found in provided data."]

    evidence_rows = []
    for idx, sentence in enumerate(evidence):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        evidence_rows.append(
            {
                "type": "signal_{}".format(idx + 1),
                "sentence": sentence,
                "score": float(np.clip(score, 0.0, 1.0)),
            }
        )

    return {
        "wallets": normalized,
        "evidence": evidence_rows,
        "evidence_sentences": evidence,
        "confidence_score": confidence,
        "linkage_strength": linkage,
    }


__all__ = ["explain_wallet_linkage"]
