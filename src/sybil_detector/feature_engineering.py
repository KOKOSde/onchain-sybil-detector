"""Feature engineering for behavioral Sybil analysis."""


from typing import Dict, Iterable, List, Optional, Set

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


def _compute_funding_signatures(tx: pd.DataFrame) -> Dict[str, str]:
    """Build per-address funding signatures by tracing through relay hops.

    The default signature is the direct dominant funding source. For relay-style
    nodes (few inbound sources, broad fanout), we trace up to 2 upstream hops so
    indirect funding chains share the same ancestor signature.
    """

    canonical = tx.sort_values("timestamp").drop_duplicates(subset=["tx_hash"], keep="first")
    if canonical.empty:
        return {}

    incoming_by_target: Dict[str, pd.DataFrame] = {
        str(addr): grp.copy()
        for addr, grp in canonical.groupby("to_addr", sort=False)
    }
    in_source_count = canonical.groupby("to_addr")["from_addr"].nunique().to_dict()
    in_tx_count = canonical.groupby("to_addr").size().to_dict()
    out_target_count = canonical.groupby("from_addr")["to_addr"].nunique().to_dict()
    out_tx_count = canonical.groupby("from_addr").size().to_dict()
    address_activity = canonical.groupby("address").size().to_dict()
    first_edge_ts = canonical.groupby(["from_addr", "to_addr"])["timestamp"].min().to_dict()
    first_incoming_ts = canonical.groupby("to_addr")["timestamp"].min().to_dict()

    def dominant_source(target: str, cutoff_ts: Optional[int] = None) -> str:
        incoming = incoming_by_target.get(str(target))
        if incoming is None or incoming.empty:
            return ""
        scoped = incoming
        if cutoff_ts is not None:
            scoped = incoming[incoming["timestamp"] <= int(cutoff_ts)]
            if scoped.empty:
                scoped = incoming
        scoped = scoped[scoped["from_addr"] != str(target)]
        if scoped.empty:
            return ""
        src_values = scoped.groupby("from_addr")["value_wei"].sum()
        if src_values.empty:
            return ""
        return str(src_values.idxmax())

    def is_relay_candidate(address: str) -> bool:
        in_sources = int(in_source_count.get(address, 0))
        in_txs = int(in_tx_count.get(address, 0))
        out_targets = int(out_target_count.get(address, 0))
        out_txs = int(out_tx_count.get(address, 0))
        return bool(in_sources <= 2 and in_txs <= 6 and out_targets >= 4 and out_txs >= 4)

    signatures: Dict[str, str] = {}
    all_addresses: Set[str] = set(canonical["address"].astype(str).tolist())
    for address in sorted(all_addresses):
        direct = dominant_source(address, cutoff_ts=int(first_incoming_ts.get(address, 0) or 0))
        signature = direct
        if int(address_activity.get(address, 0)) < 6:
            signatures[address] = signature or direct
            continue
        hop_source = direct
        cutoff_ts = int(first_incoming_ts.get(address, 0) or 0)
        visited: Set[str] = {address}

        for _ in range(2):
            if not hop_source or hop_source in visited or not is_relay_candidate(hop_source):
                break
            visited.add(hop_source)
            parent = dominant_source(hop_source, cutoff_ts=cutoff_ts if cutoff_ts > 0 else None)
            if not parent or parent in visited:
                break
            signature = parent
            edge_ts = first_edge_ts.get((parent, hop_source))
            if edge_ts is not None:
                cutoff_ts = int(edge_ts)
            hop_source = parent

        signatures[address] = signature or direct

    return signatures


def _build_address_features(
    address: str,
    group: pd.DataFrame,
    funding_signatures: Dict[str, str],
) -> Dict[str, object]:
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
        "common_funder_address": funding_signatures.get(address, top_source),
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

    funding_signatures = _compute_funding_signatures(tx)

    rows = []
    for address, group in tx.groupby("address", sort=True):
        rows.append(_build_address_features(address, group, funding_signatures))

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
