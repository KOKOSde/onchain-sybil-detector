"""Multi-chain configuration and cross-chain coordination utilities."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChainConfig:
    name: str
    aliases: List[str]
    explorer_api_base: str
    block_time_sec: float
    gas_profile: str


CHAIN_REGISTRY: Dict[str, ChainConfig] = {
    "ethereum": ChainConfig(
        name="ethereum",
        aliases=["eth", "ethereum", "mainnet"],
        explorer_api_base="https://api.etherscan.io/api",
        block_time_sec=12.0,
        gas_profile="Broad gas distribution, frequent priority fee spikes.",
    ),
    "base": ChainConfig(
        name="base",
        aliases=["base"],
        explorer_api_base="https://api.basescan.org/api",
        block_time_sec=2.0,
        gas_profile="Low median gas with occasional bursty spikes around sequencer load.",
    ),
    "bnb": ChainConfig(
        name="bnb",
        aliases=["bnb", "bsc", "bnbchain"],
        explorer_api_base="https://api.bscscan.com/api",
        block_time_sec=3.0,
        gas_profile="Generally stable gas; synchronized bot activity common in farming windows.",
    ),
    "arbitrum": ChainConfig(
        name="arbitrum",
        aliases=["arbitrum", "arb"],
        explorer_api_base="https://api.arbiscan.io/api",
        block_time_sec=0.26,
        gas_profile="Very low/flat gas, bursts around batch posting windows.",
    ),
    "optimism": ChainConfig(
        name="optimism",
        aliases=["optimism", "op"],
        explorer_api_base="https://api-optimistic.etherscan.io/api",
        block_time_sec=2.0,
        gas_profile="Low baseline gas; transaction sequencing can expose coordinated automation.",
    ),
    "polygon": ChainConfig(
        name="polygon",
        aliases=["polygon", "matic"],
        explorer_api_base="https://api.polygonscan.com/api",
        block_time_sec=2.1,
        gas_profile="Low gas chain with periodic contention spikes from campaign activity.",
    ),
}


def normalize_chain_name(chain: str) -> str:
    """Return canonical chain key for supported EVM chains."""

    value = str(chain or "").strip().lower()
    for key, cfg in CHAIN_REGISTRY.items():
        if value == key or value in cfg.aliases:
            return key
    raise ValueError("Unsupported chain '{}'. Supported: {}".format(chain, sorted(CHAIN_REGISTRY.keys())))


def get_chain_config(chain: str) -> ChainConfig:
    """Fetch canonical chain configuration."""

    return CHAIN_REGISTRY[normalize_chain_name(chain)]


def list_supported_chains() -> List[str]:
    """List supported canonical chain names."""

    return sorted(CHAIN_REGISTRY.keys())


def detect_cross_chain_coordination(
    transactions: pd.DataFrame,
    address_col: str = "address",
    chain_col: str = "chain",
    timestamp_col: str = "timestamp",
    gas_col: str = "gas_price",
    from_col: str = "from_addr",
) -> Dict[str, object]:
    """Detect cross-chain coordination signals from multi-chain transactions.

    Expected columns: address, chain, timestamp, gas_price, from_addr.
    Extra columns are ignored.
    """

    if transactions.empty:
        return {
            "bridge_timing_correlation": 0.0,
            "mirrored_gas_behavior": 0.0,
            "synchronized_windows": 0.0,
            "wallet_generation_pattern": 0.0,
            "common_funding_sources": 0.0,
            "coordination_score": 0.0,
            "evidence": [],
        }

    tx = transactions.copy()
    for col in [address_col, chain_col, from_col]:
        tx[col] = tx[col].astype(str).str.lower()
    tx[timestamp_col] = pd.to_numeric(tx[timestamp_col], errors="coerce").fillna(0).astype(np.int64)
    tx[gas_col] = pd.to_numeric(tx[gas_col], errors="coerce").fillna(0.0)

    # Keep only addresses observed on at least two chains for cross-chain scoring.
    chain_counts = tx.groupby(address_col)[chain_col].nunique()
    multi_chain_wallets = chain_counts[chain_counts >= 2].index.tolist()
    cross = tx[tx[address_col].isin(multi_chain_wallets)].copy()
    if cross.empty:
        return {
            "bridge_timing_correlation": 0.0,
            "mirrored_gas_behavior": 0.0,
            "synchronized_windows": 0.0,
            "wallet_generation_pattern": 0.0,
            "common_funding_sources": 0.0,
            "coordination_score": 0.0,
            "evidence": ["No wallets active on multiple supported chains."],
        }

    evidence: List[str] = []

    # 1) Bridge timing correlation: min pairwise cross-chain first-seen delta.
    first_seen = cross.groupby([address_col, chain_col])[timestamp_col].min().reset_index()
    deltas = []
    for _, group in first_seen.groupby(address_col):
        times = sorted(group[timestamp_col].tolist())
        if len(times) < 2:
            continue
        deltas.extend(np.diff(times))
    bridge_timing = float(np.mean(np.array(deltas) <= 600)) if deltas else 0.0
    if bridge_timing > 0:
        evidence.append("{} of multi-chain wallets have <=10 minute first-hop deltas.".format(round(bridge_timing, 3)))

    # 2) Mirrored gas behavior across chains.
    gas_stats = cross.groupby([address_col, chain_col])[gas_col].median().unstack(fill_value=np.nan)
    gas_cv = gas_stats.apply(lambda row: float(np.nanstd(row) / np.nanmean(row)) if np.nanmean(row) > 0 else 1.0, axis=1)
    mirrored_gas = float(np.mean(gas_cv <= 0.15)) if len(gas_cv) else 0.0
    if mirrored_gas > 0:
        evidence.append("{} of multi-chain wallets show mirrored median gas behavior (CV<=0.15).".format(round(mirrored_gas, 3)))

    # 3) Synchronized windows across chains via active-hour overlap.
    cross["hour"] = pd.to_datetime(cross[timestamp_col], unit="s", utc=True).dt.hour
    hour_counts = (
        cross.groupby([address_col, chain_col, "hour"])
        .size()
        .reset_index(name="count")
    )

    sync_scores = []
    for _, wallet_rows in hour_counts.groupby(address_col):
        vectors = []
        for _, chain_rows in wallet_rows.groupby(chain_col):
            vec = np.zeros(24, dtype=float)
            for _, row in chain_rows.iterrows():
                vec[int(row["hour"])] = float(row["count"])
            total = float(vec.sum())
            if total > 0:
                vec = vec / total
            vectors.append(vec)

        if len(vectors) < 2:
            continue

        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                a = vectors[i]
                b = vectors[j]
                denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
                sims.append(float(np.dot(a, b) / denom))
        if sims:
            sync_scores.append(float(np.mean(sims)))
    synchronized_windows = float(np.mean(sync_scores)) if sync_scores else 0.0
    if synchronized_windows > 0:
        evidence.append("Average cross-chain active-hour cosine similarity: {:.3f}.".format(synchronized_windows))

    # 4) Repeated wallet-generation patterns (shared leading prefix entropy drop).
    prefixes = tx[address_col].str.slice(2, 10)
    top_prefix_freq = prefixes.value_counts(normalize=True).iloc[0] if not prefixes.empty else 0.0
    wallet_generation = float(top_prefix_freq)
    if wallet_generation > 0.03:
        evidence.append("High prefix concentration detected (top 8-hex prefix share {:.3f}).".format(wallet_generation))

    # 5) Common funding sources across chains.
    if from_col in cross.columns:
        funder_cross = cross.groupby(from_col)[chain_col].nunique()
        common_funding = float(np.mean(funder_cross >= 2)) if len(funder_cross) else 0.0
    else:
        common_funding = 0.0
    if common_funding > 0:
        evidence.append("{} of funding addresses fund wallets across >=2 chains.".format(round(common_funding, 3)))

    coordination_score = float(
        np.clip(
            0.25 * bridge_timing
            + 0.2 * mirrored_gas
            + 0.2 * synchronized_windows
            + 0.15 * wallet_generation
            + 0.2 * common_funding,
            0.0,
            1.0,
        )
    )

    return {
        "bridge_timing_correlation": bridge_timing,
        "mirrored_gas_behavior": mirrored_gas,
        "synchronized_windows": synchronized_windows,
        "wallet_generation_pattern": wallet_generation,
        "common_funding_sources": common_funding,
        "coordination_score": coordination_score,
        "evidence": evidence,
        "multi_chain_wallet_count": int(len(multi_chain_wallets)),
    }


def attach_chain_defaults(transactions: pd.DataFrame, chain: str) -> pd.DataFrame:
    """Attach canonical chain column and block-time hints to a transaction dataframe."""

    cfg = get_chain_config(chain)
    df = transactions.copy()
    df["chain"] = cfg.name
    df["chain_block_time_sec"] = float(cfg.block_time_sec)
    df["chain_gas_profile"] = cfg.gas_profile
    return df


__all__ = [
    "ChainConfig",
    "CHAIN_REGISTRY",
    "normalize_chain_name",
    "get_chain_config",
    "list_supported_chains",
    "detect_cross_chain_coordination",
    "attach_chain_defaults",
]
