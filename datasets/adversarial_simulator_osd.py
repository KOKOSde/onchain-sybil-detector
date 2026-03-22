"""Adversarial Sybil simulation utilities for OSD difficulty benchmarking."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from datasets.synthetic_generator import generate_synthetic_sybil_network  # type: ignore
except Exception:
    _SYNTH_PATH = Path(__file__).resolve().parent / "synthetic_generator.py"
    _SYNTH_SPEC = importlib_util.spec_from_file_location("osd_synthetic_generator", _SYNTH_PATH)
    if _SYNTH_SPEC is None or _SYNTH_SPEC.loader is None:
        raise ImportError(f"Unable to load synthetic generator from {_SYNTH_PATH}")
    _SYNTH_MODULE = importlib_util.module_from_spec(_SYNTH_SPEC)
    sys.modules[_SYNTH_SPEC.name] = _SYNTH_MODULE
    _SYNTH_SPEC.loader.exec_module(_SYNTH_MODULE)
    generate_synthetic_sybil_network = _SYNTH_MODULE.generate_synthetic_sybil_network

TX_COLUMNS = [
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

EVM_CHAINS = ["ethereum", "base", "bnb", "arbitrum", "optimism", "polygon"]


@dataclass
class _TxAppender:
    rows: List[Dict[str, object]]
    base_block: int
    counter: int

    def append_transfer(
        self,
        from_addr: str,
        to_addr: str,
        timestamp: int,
        value_wei: int,
        gas_price: int,
        gas_used: int,
        input_data_prefix: str,
        chain: str,
    ) -> None:
        tx_hash = f"0xadv{self.counter:060x}"[-66:]
        block_number = self.base_block + max(0, int((timestamp - 1_700_000_000) // 12))
        base = {
            "tx_hash": tx_hash,
            "block_number": int(block_number),
            "timestamp": int(timestamp),
            "from_addr": from_addr.lower(),
            "to_addr": to_addr.lower(),
            "value_wei": int(max(0, value_wei)),
            "gas_price": int(max(1, gas_price)),
            "gas_used": int(max(21_000, gas_used)),
            "input_data_prefix": str(input_data_prefix),
            "chain": chain,
        }
        self.rows.append({"address": from_addr.lower(), **base})
        self.rows.append({"address": to_addr.lower(), **base})
        self.counter += 1


def _normalize_levels(difficulty: int, enabled_levels: Optional[Sequence[int]]) -> List[int]:
    if enabled_levels is not None:
        levels = sorted({int(x) for x in enabled_levels if 1 <= int(x) <= 8})
        if not levels:
            raise ValueError("enabled_levels must contain values from 1..8")
        return levels
    level = int(difficulty)
    if level < 1 or level > 8:
        raise ValueError("difficulty must be in [1, 8]")
    return [level]


def _group_sybil_members(labels: pd.DataFrame) -> Dict[int, List[str]]:
    sybil = labels[labels["is_sybil"] == 1].copy()
    grouped: Dict[int, List[str]] = {}
    for cluster_id, group in sybil.groupby("cluster_id"):
        grouped[int(cluster_id)] = group["address"].astype(str).str.lower().tolist()
    return grouped


def _assign_difficulty_metadata(labels: pd.DataFrame, levels: List[int]) -> pd.DataFrame:
    out = labels.copy()
    tags = [f"L{lvl}" for lvl in levels]
    out["difficulty_level"] = max(levels)
    out["difficulty_tags"] = ",".join(tags)
    return out


def _ensure_chain_column(transactions: pd.DataFrame) -> pd.DataFrame:
    out = transactions.copy()
    if "chain" not in out.columns:
        out["chain"] = "ethereum"
    out["chain"] = out["chain"].astype(str).str.lower()
    return out


def _apply_level_2_timing_jitter(
    transactions: pd.DataFrame,
    sybil_addresses: set,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = transactions.copy()
    mask = out["address"].isin(sybil_addresses)
    jitter = rng.integers(-1800, 1801, size=int(mask.sum()))
    out.loc[mask, "timestamp"] = (out.loc[mask, "timestamp"].astype(int) + jitter).clip(lower=1)
    return out


def _apply_level_3_indirect_funding(
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = transactions.copy()
    out["address"] = out["address"].astype(str).str.lower()
    out["from_addr"] = out["from_addr"].astype(str).str.lower()
    out["to_addr"] = out["to_addr"].astype(str).str.lower()

    sybil_members = _group_sybil_members(labels)
    direct_hashes = set()

    appender = _TxAppender(rows=[], base_block=int(out["block_number"].min()), counter=1)
    for cluster_id, members in sybil_members.items():
        funder = f"0x{(0xF10000 + cluster_id):040x}"
        relay_a = f"0x{(0xE10000 + cluster_id * 3):040x}"
        relay_b = f"0x{(0xE10000 + cluster_id * 3 + 1):040x}"

        member_rows = out[(out["to_addr"].isin(members)) & (~out["from_addr"].isin(members))]
        if not member_rows.empty:
            direct_hashes.update(member_rows["tx_hash"].astype(str).tolist())

        for member in members:
            member_tx = out[(out["address"] == member) & (out["timestamp"] > 0)]
            first_ts = int(member_tx["timestamp"].min()) if not member_tx.empty else 1_700_000_000
            base_ts = first_ts - int(rng.integers(600, 3600))
            funding = int(rng.normal(3.2e17, 4.5e16))
            gas_price = int(rng.normal(21e9, 2e9))

            appender.append_transfer(
                from_addr=funder,
                to_addr=relay_a,
                timestamp=base_ts,
                value_wei=funding,
                gas_price=gas_price,
                gas_used=46_000,
                input_data_prefix="0x",
                chain="ethereum",
            )
            appender.append_transfer(
                from_addr=relay_a,
                to_addr=relay_b,
                timestamp=base_ts + int(rng.integers(90, 420)),
                value_wei=int(funding * 0.98),
                gas_price=gas_price,
                gas_used=48_000,
                input_data_prefix="0x",
                chain="ethereum",
            )
            appender.append_transfer(
                from_addr=relay_b,
                to_addr=member,
                timestamp=base_ts + int(rng.integers(460, 1200)),
                value_wei=int(funding * 0.95),
                gas_price=gas_price,
                gas_used=51_000,
                input_data_prefix="0x",
                chain="ethereum",
            )

    if direct_hashes:
        out = out[~out["tx_hash"].isin(direct_hashes)].copy()

    if appender.rows:
        out = pd.concat([out, pd.DataFrame(appender.rows)], ignore_index=True)
    return out


def _apply_level_4_mixed_gas(
    transactions: pd.DataFrame,
    sybil_addresses: set,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = transactions.copy()
    mask = out["address"].isin(sybil_addresses)
    n = int(mask.sum())
    if n == 0:
        return out
    gas_mult = rng.uniform(0.6, 1.6, size=n)
    used_mult = rng.uniform(0.8, 1.35, size=n)
    out.loc[mask, "gas_price"] = (out.loc[mask, "gas_price"].astype(float) * gas_mult).astype(int)
    out.loc[mask, "gas_used"] = (out.loc[mask, "gas_used"].astype(float) * used_mult).astype(int)
    return out


def _apply_level_5_cluster_splitting(labels: pd.DataFrame, wallets_per_cluster: int) -> pd.DataFrame:
    out = labels.copy()
    sybil = out["is_sybil"] == 1
    for cluster_id, group in out[sybil].groupby("cluster_id"):
        members = group.sort_values("address").index.tolist()
        split_size = max(1, min(5, wallets_per_cluster // 4))
        for i, idx in enumerate(members):
            subgroup = i // split_size
            out.loc[idx, "cluster_id"] = int(cluster_id) * 10 + int(subgroup)
    return out


def _apply_level_6_chain_hopping(
    transactions: pd.DataFrame,
    sybil_addresses: set,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = _ensure_chain_column(transactions)
    mask = out["address"].isin(sybil_addresses)
    assigned = rng.choice(EVM_CHAINS, size=int(mask.sum()), p=[0.45, 0.12, 0.12, 0.11, 0.1, 0.1])
    out.loc[mask, "chain"] = assigned
    return out


def _apply_level_7_burner_wallets(
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_tx = _ensure_chain_column(transactions)
    out_labels = labels.copy()

    appender = _TxAppender(rows=[], base_block=int(out_tx["block_number"].min()), counter=800000)
    new_labels = []

    sybil_members = _group_sybil_members(out_labels)
    for cluster_id, members in sybil_members.items():
        take = max(1, len(members) // 6)
        selected = rng.choice(members, size=take, replace=False)
        for i, member in enumerate(selected):
            burner = f"0x{(0xB70000 + cluster_id * 100 + i):040x}"
            member_rows = out_tx[(out_tx["address"] == member) & (out_tx["timestamp"] > 0)]
            t0 = int(member_rows["timestamp"].min()) if not member_rows.empty else 1_700_000_000
            chain = (
                str(member_rows["chain"].mode().iloc[0])
                if "chain" in member_rows.columns and not member_rows.empty
                else "ethereum"
            )

            appender.append_transfer(
                from_addr=member,
                to_addr=burner,
                timestamp=t0 + int(rng.integers(500, 2800)),
                value_wei=int(rng.normal(7e16, 9e15)),
                gas_price=int(rng.normal(20e9, 2.5e9)),
                gas_used=52_000,
                input_data_prefix="0xa9059cbb",
                chain=chain,
            )
            appender.append_transfer(
                from_addr=burner,
                to_addr=f"0x{(0xC80000 + cluster_id):040x}",
                timestamp=t0 + int(rng.integers(3000, 6200)),
                value_wei=int(rng.normal(6.6e16, 9e15)),
                gas_price=int(rng.normal(20e9, 2.5e9)),
                gas_used=52_000,
                input_data_prefix="0x23b872dd",
                chain=chain,
            )

            new_labels.append(
                {
                    "address": burner,
                    "is_sybil": 1,
                    "cluster_id": int(cluster_id),
                    "operator_id": int(out_labels[out_labels["cluster_id"] == cluster_id]["operator_id"].iloc[0]),
                    "difficulty_level": out_labels.get("difficulty_level", pd.Series([7])).iloc[0],
                    "difficulty_tags": "L7",
                }
            )

    if appender.rows:
        out_tx = pd.concat([out_tx, pd.DataFrame(appender.rows)], ignore_index=True)
    if new_labels:
        out_labels = pd.concat([out_labels, pd.DataFrame(new_labels)], ignore_index=True)
    out_labels = out_labels.drop_duplicates(subset=["address"]).reset_index(drop=True)
    return out_tx, out_labels


def _apply_level_8_delayed_coordination(
    transactions: pd.DataFrame,
    sybil_addresses: set,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = transactions.copy()
    shifts: Dict[str, int] = {}
    for address in sorted(sybil_addresses):
        shifts[address] = int(rng.integers(0, 6 * 86_400))
    mask = out["address"].isin(sybil_addresses)
    out.loc[mask, "timestamp"] = out.loc[mask].apply(
        lambda row: int(row["timestamp"]) + int(shifts.get(str(row["address"]), 0)), axis=1
    )
    return out


def _finalize_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    out = transactions.copy()
    if "chain" not in out.columns:
        out["chain"] = "ethereum"
    out["chain"] = out["chain"].astype(str).str.lower()

    for col in TX_COLUMNS:
        if col not in out.columns:
            out[col] = 0 if col not in {"address", "tx_hash", "from_addr", "to_addr", "input_data_prefix"} else ""

    for col in ["timestamp", "block_number", "value_wei", "gas_price", "gas_used"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    out["address"] = out["address"].astype(str).str.lower()
    out["from_addr"] = out["from_addr"].astype(str).str.lower()
    out["to_addr"] = out["to_addr"].astype(str).str.lower()
    out["tx_hash"] = out["tx_hash"].astype(str)
    out["input_data_prefix"] = out["input_data_prefix"].astype(str).str.slice(0, 10)

    out = out.sort_values(["timestamp", "block_number", "tx_hash", "address"]).reset_index(drop=True)
    cols = TX_COLUMNS + [c for c in out.columns if c not in TX_COLUMNS]
    return out[cols]


def generate_adversarial_sybils(
    difficulty: int = 1,
    num_clusters: int = 10,
    wallets_per_cluster: int = 12,
    num_legit: int = 300,
    seed: int = 42,
    enabled_levels: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate adversarial sybil data with independently toggleable levels.

    Args:
        difficulty: convenience selector in [1, 8] when enabled_levels is None.
        enabled_levels: optional explicit list of enabled levels from 1..8.
    """

    levels = _normalize_levels(difficulty=difficulty, enabled_levels=enabled_levels)
    rng = np.random.default_rng(seed)

    transactions, labels = generate_synthetic_sybil_network(
        num_legit=num_legit,
        num_sybil_clusters=num_clusters,
        addrs_per_cluster=wallets_per_cluster,
        seed=seed,
    )

    labels = _assign_difficulty_metadata(labels, levels)
    tx = _ensure_chain_column(transactions)
    sybil_addresses = set(labels.loc[labels["is_sybil"] == 1, "address"].astype(str).str.lower())

    for lvl in levels:
        if lvl == 1:
            continue
        if lvl == 2:
            tx = _apply_level_2_timing_jitter(tx, sybil_addresses, rng)
        elif lvl == 3:
            tx = _apply_level_3_indirect_funding(tx, labels, rng)
        elif lvl == 4:
            tx = _apply_level_4_mixed_gas(tx, sybil_addresses, rng)
        elif lvl == 5:
            labels = _apply_level_5_cluster_splitting(labels, wallets_per_cluster)
        elif lvl == 6:
            tx = _apply_level_6_chain_hopping(tx, sybil_addresses, rng)
        elif lvl == 7:
            tx, labels = _apply_level_7_burner_wallets(tx, labels, rng)
            sybil_addresses = set(labels.loc[labels["is_sybil"] == 1, "address"].astype(str).str.lower())
        elif lvl == 8:
            tx = _apply_level_8_delayed_coordination(tx, sybil_addresses, rng)

    tx = _finalize_transactions(tx)
    labels = labels.drop_duplicates(subset=["address"]).sort_values("address").reset_index(drop=True)
    labels["difficulty_level"] = max(levels)
    labels["difficulty_tags"] = labels.get("difficulty_tags", "")

    return tx, labels


def run_adversarial_detection_benchmark(
    detector_factory: Callable[[], object],
    levels: Iterable[int] = tuple(range(1, 9)),
    num_clusters: int = 10,
    wallets_per_cluster: int = 12,
    num_legit: int = 300,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Run detector over levels and return address/cluster detection metrics."""

    from sybil_detector.benchmark import run_benchmark

    results: Dict[str, Dict[str, float]] = {}
    for level in levels:
        tx, labels = generate_adversarial_sybils(
            difficulty=int(level),
            num_clusters=num_clusters,
            wallets_per_cluster=wallets_per_cluster,
            num_legit=num_legit,
            seed=seed,
        )
        detector = detector_factory()
        metrics = run_benchmark(detector, tx, labels)
        results[f"level_{int(level)}"] = {
            "address_precision": float(metrics["address"]["precision"]),
            "address_recall": float(metrics["address"]["recall"]),
            "address_f1": float(metrics["address"]["f1"]),
            "cluster_precision": float(metrics["cluster"]["precision"]),
            "cluster_recall": float(metrics["cluster"]["recall"]),
            "cluster_f1": float(metrics["cluster"]["f1"]),
            "n_predicted_sybil_addresses": float(metrics["n_predicted_sybil_addresses"]),
        }
    return results


__all__ = ["generate_adversarial_sybils", "run_adversarial_detection_benchmark"]
