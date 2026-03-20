"""Feature engineering for behavioral Sybil analysis."""


from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

REQUIRED_TRANSACTION_COLUMNS = [
    "address",
    "tx_hash",
    "block_number",
    "timestamp",
    "from_addr",
    "to_addr",
    "value_wei",
    "gas_price",
    "gas_used",
    "input_data_prefix",
]

FEATURE_COLUMNS = [
    "hour_of_day_entropy",
    "day_of_week_entropy",
    "median_inter_tx_time_sec",
    "std_inter_tx_time_sec",
    "burst_ratio",
    "first_tx_timestamp",
    "active_days",
    "activity_span_days",
    "median_gas_price",
    "gas_price_cv",
    "gas_price_mode_ratio",
    "median_gas_used",
    "gas_efficiency",
    "median_value_wei",
    "value_entropy",
    "round_number_ratio",
    "unique_value_count",
    "unique_counterparties",
    "in_degree",
    "out_degree",
    "self_loop_count",
    "common_funder_address",
    "pct_funds_from_top_source",
    "time_to_first_outgoing_sec",
    "funding_source_count",
]

REQUIRED_FEATURES = FEATURE_COLUMNS.copy()
HOUR_HIST_COLUMNS = [f"hour_hist_{i:02d}" for i in range(24)]


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    entropy = -(probs * np.log2(probs)).sum()
    return float(entropy)


def _coefficient_of_variation(values: np.ndarray) -> float:
    values = values.astype(float)
    if values.size == 0:
        return 0.0
    mean_value = float(values.mean())
    if mean_value == 0:
        return 0.0
    return float(values.std(ddof=0) / mean_value)


def _mode_ratio(values: Iterable[float], round_to: int = 0) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    rounded = np.round(arr, round_to)
    _, counts = np.unique(rounded, return_counts=True)
    return float(counts.max() / arr.size)


def _value_entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if np.all(values == values[0]):
        return 0.0
    edges = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges = np.unique(edges)
    if edges.size < 2:
        return 0.0
    counts, _ = np.histogram(values, bins=edges)
    return _entropy_from_counts(counts)


def _hour_histogram(timestamps: pd.Series) -> List[float]:
    dt = pd.to_datetime(timestamps, unit="s", utc=True)
    counts = np.bincount(dt.dt.hour.values, minlength=24).astype(float)
    total = counts.sum()
    if total == 0:
        return [0.0] * 24
    return (counts / total).tolist()


def _required_columns_present(transactions: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_TRANSACTION_COLUMNS if c not in transactions.columns]
    if missing:
        raise ValueError(f"transactions is missing required columns: {missing}")


def _build_address_features(address: str, group: pd.DataFrame) -> Dict[str, object]:
    group = group.sort_values("timestamp").copy()
    timestamps = group["timestamp"].astype(int).to_numpy()
    values = group["value_wei"].astype(float).to_numpy()
    gas_price = group["gas_price"].astype(float).to_numpy()
    gas_used = group["gas_used"].astype(float).to_numpy()

    inter_times = np.diff(timestamps)
    first_ts = int(timestamps.min()) if timestamps.size else 0
    last_ts = int(timestamps.max()) if timestamps.size else first_ts

    dt = pd.to_datetime(group["timestamp"], unit="s", utc=True)
    hour_counts = np.bincount(dt.dt.hour.values, minlength=24)
    dow_counts = np.bincount(dt.dt.dayofweek.values, minlength=7)

    is_outgoing = group["from_addr"] == address
    is_incoming = group["to_addr"] == address
    incoming = group[is_incoming]
    outgoing = group[is_outgoing]

    counterparties = np.where(
        is_outgoing,
        group["to_addr"].to_numpy(dtype=str),
        group["from_addr"].to_numpy(dtype=str),
    )
    counterparties = counterparties[counterparties != address]

    incoming_sources = incoming["from_addr"].value_counts()
    total_incoming_value = float(incoming["value_wei"].sum())
    top_source = ""
    top_source_ratio = 0.0
    if not incoming_sources.empty:
        top_source = str(incoming_sources.idxmax())
        top_source_value = float(incoming.loc[incoming["from_addr"] == top_source, "value_wei"].sum())
        if total_incoming_value > 0:
            top_source_ratio = top_source_value / total_incoming_value

    first_in_ts = int(incoming["timestamp"].min()) if not incoming.empty else first_ts
    first_out_ts = int(outgoing["timestamp"].min()) if not outgoing.empty else first_ts
    time_to_first_out = float(max(0, first_out_ts - first_in_ts))

    gas_denom = gas_price * np.maximum(gas_used, 1.0)
    gas_efficiency = np.divide(values, gas_denom, out=np.zeros_like(values), where=gas_denom > 0)

    round_ratio = float(np.mean((values.astype(np.int64) % int(1e15)) == 0)) if values.size else 0.0

    row: Dict[str, object] = {
        "address": address,
        "hour_of_day_entropy": _entropy_from_counts(hour_counts),
        "day_of_week_entropy": _entropy_from_counts(dow_counts),
        "median_inter_tx_time_sec": float(np.median(inter_times)) if inter_times.size else 0.0,
        "std_inter_tx_time_sec": float(np.std(inter_times)) if inter_times.size else 0.0,
        "burst_ratio": float(np.mean(inter_times <= 3600)) if inter_times.size else 0.0,
        "first_tx_timestamp": float(first_ts),
        "active_days": float(dt.dt.date.nunique()),
        "activity_span_days": float(max(0, (last_ts - first_ts) / 86_400.0)),
        "median_gas_price": float(np.median(gas_price)) if gas_price.size else 0.0,
        "gas_price_cv": _coefficient_of_variation(gas_price),
        "gas_price_mode_ratio": _mode_ratio(gas_price, round_to=-7),
        "median_gas_used": float(np.median(gas_used)) if gas_used.size else 0.0,
        "gas_efficiency": float(np.median(gas_efficiency)) if gas_efficiency.size else 0.0,
        "median_value_wei": float(np.median(values)) if values.size else 0.0,
        "value_entropy": _value_entropy(values),
        "round_number_ratio": round_ratio,
        "unique_value_count": float(np.unique(values).size),
        "unique_counterparties": float(np.unique(counterparties).size),
        "in_degree": float(incoming["from_addr"].nunique()),
        "out_degree": float(outgoing["to_addr"].nunique()),
        "self_loop_count": float(((group["from_addr"] == address) & (group["to_addr"] == address)).sum()),
        "common_funder_address": top_source,
        "pct_funds_from_top_source": float(top_source_ratio),
        "time_to_first_outgoing_sec": time_to_first_out,
        "funding_source_count": float(incoming["from_addr"].nunique()),
        # Auxiliary vector for temporal coordination computations.
        "hour_histogram": _hour_histogram(group["timestamp"]),
    }
    return row


def extract_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Extract 25 address-level behavioral features from transaction records."""

    _required_columns_present(transactions)

    if transactions.empty:
        return pd.DataFrame(columns=["address", *FEATURE_COLUMNS, "hour_histogram"])

    tx = transactions.copy()
    for col in ["address", "from_addr", "to_addr", "tx_hash", "input_data_prefix"]:
        tx[col] = tx[col].astype(str).str.lower()
    tx["timestamp"] = pd.to_numeric(tx["timestamp"], errors="coerce").fillna(0).astype(int)
    tx["value_wei"] = pd.to_numeric(tx["value_wei"], errors="coerce").fillna(0).astype(float)
    tx["gas_price"] = pd.to_numeric(tx["gas_price"], errors="coerce").fillna(0).astype(float)
    tx["gas_used"] = pd.to_numeric(tx["gas_used"], errors="coerce").fillna(0).astype(float)

    rows = []
    for address, group in tx.groupby("address", sort=True):
        rows.append(_build_address_features(address, group))

    features = pd.DataFrame(rows)
    numeric_cols = [c for c in FEATURE_COLUMNS if c != "common_funder_address"]
    features[numeric_cols] = features[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    features["common_funder_address"] = features["common_funder_address"].fillna("").astype(str)

    hour_matrix = np.vstack(
        features["hour_histogram"]
        .apply(lambda vals: vals if isinstance(vals, list) and len(vals) == 24 else [0.0] * 24)
        .to_numpy()
    )
    for idx, col in enumerate(HOUR_HIST_COLUMNS):
        features[col] = hour_matrix[:, idx]

    ordered_columns = ["address", *FEATURE_COLUMNS, *HOUR_HIST_COLUMNS, "hour_histogram"]
    return features[ordered_columns].sort_values("address").reset_index(drop=True)
