"""Synthetic on-chain transaction generator for Sybil detection experiments."""


from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

METHOD_PREFIXES = ["0xa9059cbb", "0x23b872dd", "0x095ea7b3", "0x"]


@dataclass
class _TxBuilder:
    seed: int
    start_ts: int
    base_block: int

    def __post_init__(self) -> None:
        self.rows: List[dict] = []
        self.counter = 0

    def add_transfer(
        self,
        from_addr: str,
        to_addr: str,
        timestamp: float,
        value_wei: float,
        gas_price: float,
        gas_used: float,
        input_data_prefix: str,
    ) -> None:
        ts = int(max(self.start_ts, timestamp))
        block_number = self.base_block + max(0, int((ts - self.start_ts) // 12))
        tx_hash = f"0x{self.seed:08x}{self.counter:056x}"
        base_row = {
            "tx_hash": tx_hash,
            "block_number": block_number,
            "timestamp": ts,
            "from_addr": from_addr.lower(),
            "to_addr": to_addr.lower(),
            "value_wei": int(max(0, value_wei)),
            "gas_price": int(max(1, gas_price)),
            "gas_used": int(max(21000, gas_used)),
            "input_data_prefix": input_data_prefix,
        }
        self.rows.append({"address": from_addr.lower(), **base_row})
        self.rows.append({"address": to_addr.lower(), **base_row})
        self.counter += 1


def _addr(num: int) -> str:
    return f"0x{num:040x}"


def _random_counterparty(rng: np.random.Generator, pool: List[str], self_addr: str) -> str:
    if not pool:
        return self_addr
    candidate = pool[int(rng.integers(0, len(pool)))]
    if candidate == self_addr:
        return _addr(int(rng.integers(10**6, 10**7)))
    return candidate


def generate_synthetic_sybil_network(
    num_legit: int = 500,
    num_sybil_clusters: int = 10,
    addrs_per_cluster: int = 20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic transactions and labels for Sybil detection benchmarks.

    Returns:
        transactions_df: one row per address-involved transfer event.
        labels_df: one row per address with Sybil and cluster labels.
    """

    rng = np.random.default_rng(seed)
    start_ts = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp())
    builder = _TxBuilder(seed=seed, start_ts=start_ts, base_block=19_000_000)

    labels: List[dict] = []
    service_pool = [_addr(0x500000 + i) for i in range(200)]

    legit_addresses = [_addr(0x100000 + i) for i in range(num_legit)]
    all_counterparties = service_pool + legit_addresses

    for i, address in enumerate(legit_addresses):
        labels.append(
            {
                "address": address,
                "is_sybil": 0,
                "cluster_id": -1,
                "operator_id": -1,
            }
        )
        n_txs = int(rng.integers(10, 35))
        current_ts = start_ts + int(rng.integers(0, 70 * 86_400))
        for _ in range(n_txs):
            current_ts += float(rng.exponential(1.6 * 86_400))
            counterparty = _random_counterparty(rng, all_counterparties, address)
            incoming = bool(rng.random() < 0.47)
            from_addr = counterparty if incoming else address
            to_addr = address if incoming else counterparty
            value_wei = float(rng.lognormal(mean=38.0, sigma=1.2))
            gas_price = float(rng.lognormal(mean=24.6, sigma=0.33))
            gas_used = float(rng.normal(loc=82_000, scale=15_000))
            if rng.random() < 0.06:
                # Occasional self-loop and noisy method signatures.
                from_addr = address
                to_addr = address
            builder.add_transfer(
                from_addr=from_addr,
                to_addr=to_addr,
                timestamp=current_ts,
                value_wei=value_wei,
                gas_price=gas_price,
                gas_used=gas_used,
                input_data_prefix=METHOD_PREFIXES[int(rng.integers(0, len(METHOD_PREFIXES)))],
            )

    for cluster_id in range(num_sybil_clusters):
        operator_id = cluster_id // 2
        funder = _addr(0xF00000 + cluster_id)
        cluster_base_hour = int(rng.integers(1, 23))
        cluster_base_gas = float(rng.normal(21e9, 1.6e9))
        cluster_start = start_ts + cluster_id * 3 * 86_400 + int(rng.integers(0, 3600))

        for j in range(addrs_per_cluster):
            address = _addr(0xA00000 + cluster_id * 1000 + j)
            labels.append(
                {
                    "address": address,
                    "is_sybil": 1,
                    "cluster_id": cluster_id,
                    "operator_id": operator_id,
                }
            )

            creation_ts = (
                cluster_start
                + j * int(rng.integers(140, 620))
                + float(rng.normal(0, 55))
            )
            funding_value = float(rng.normal(3.4e17, 4.8e16))
            builder.add_transfer(
                from_addr=funder,
                to_addr=address,
                timestamp=creation_ts,
                value_wei=funding_value,
                gas_price=cluster_base_gas * (1.0 + float(rng.normal(0, 0.02))),
                gas_used=float(rng.normal(loc=54_500, scale=1_800)),
                input_data_prefix="0x",
            )

            if rng.random() < 0.18:
                # Decoy source to avoid trivial same-funder heuristics.
                builder.add_transfer(
                    from_addr=_random_counterparty(rng, service_pool, address),
                    to_addr=address,
                    timestamp=creation_ts + int(rng.integers(60, 1500)),
                    value_wei=float(rng.normal(6.0e16, 1.2e16)),
                    gas_price=cluster_base_gas * (1.0 + float(rng.normal(0, 0.07))),
                    gas_used=float(rng.normal(loc=57_500, scale=2_300)),
                    input_data_prefix="0xa9059cbb",
                )

            n_actions = int(rng.integers(7, 16))
            for k in range(n_actions):
                day_offset = int(rng.integers(0, 18))
                correlated_hour = (cluster_base_hour + int(rng.integers(-1, 2))) % 24
                burst_jitter = float(rng.normal(0, 640))
                action_ts = (
                    creation_ts
                    + day_offset * 86_400
                    + correlated_hour * 3600
                    + burst_jitter
                )

                cp = _addr(0xC00000 + cluster_id * 300 + int(rng.integers(0, 220)))
                outgoing = bool(rng.random() < 0.72)
                from_addr = address if outgoing else cp
                to_addr = cp if outgoing else address

                gas_noise = float(rng.normal(0, 0.018))
                if rng.random() < 0.12:
                    gas_noise += float(rng.normal(0, 0.18))
                gas_price = cluster_base_gas * (1.0 + gas_noise)

                value_wei = float(rng.normal(8.6e16, 2.2e16))
                if rng.random() < 0.27:
                    value_wei = round(value_wei / 1e15) * 1e15

                builder.add_transfer(
                    from_addr=from_addr,
                    to_addr=to_addr,
                    timestamp=action_ts,
                    value_wei=value_wei,
                    gas_price=gas_price,
                    gas_used=float(rng.normal(loc=51_500, scale=2_600)),
                    input_data_prefix=METHOD_PREFIXES[int(rng.integers(0, len(METHOD_PREFIXES)))],
                )

            if rng.random() < 0.16:
                # Add occasional off-pattern transfer as camouflage.
                builder.add_transfer(
                    from_addr=address,
                    to_addr=_random_counterparty(rng, service_pool, address),
                    timestamp=creation_ts + int(rng.integers(12 * 86_400, 28 * 86_400)),
                    value_wei=float(rng.normal(2.4e17, 4.0e16)),
                    gas_price=float(rng.lognormal(mean=24.9, sigma=0.5)),
                    gas_used=float(rng.normal(loc=73_000, scale=11_000)),
                    input_data_prefix="0x095ea7b3",
                )

    transactions_df = pd.DataFrame(builder.rows)
    if transactions_df.empty:
        transactions_df = pd.DataFrame(
            columns=[
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
        )
    transactions_df = transactions_df.sort_values(
        ["timestamp", "block_number", "tx_hash", "address"]
    ).reset_index(drop=True)

    labels_df = pd.DataFrame(labels).drop_duplicates("address").sort_values("address").reset_index(
        drop=True
    )

    return transactions_df, labels_df
