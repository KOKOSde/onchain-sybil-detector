"""On-chain data ingestion with SQLite caching and explicit offline mode."""


import importlib.util
import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ETHERSCAN_KEY = os.getenv("ETHERSCAN_API_KEY")  # None if not set = still works
ALCHEMY_KEY = os.getenv("ALCHEMY_API_KEY")
RPC_URL = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}" if ALCHEMY_KEY else "https://rpc.ankr.com/eth"

CHAIN_ID_MAP = {
    "eth": 1,
    "ethereum": 1,
    "base": 8453,
    "bnb": 56,
    "bsc": 56,
    "arbitrum": 42161,
    "arb": 42161,
    "optimism": 10,
    "op": 10,
    "polygon": 137,
    "matic": 137,
}


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

TOKEN_TRANSFER_COLUMNS = [
    "address",
    "tx_hash",
    "timestamp",
    "from_addr",
    "to_addr",
    "token_symbol",
    "token_value",
]


class ChainDataFetcher(object):
    """Fetch and cache transaction activity for target addresses.

    Behavior:
    - online mode: reads from cache, then fetches missing addresses via Etherscan V2
      and raises clear errors on API/network failures
    - offline mode: reads from cache, then fills missing addresses from synthetic data
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        etherscan_key: Optional[str] = None,
        cache_dir: str = "cache",
        chain: str = "ethereum",
        offline: bool = False,
    ) -> None:
        self.rpc_url = rpc_url or RPC_URL
        self.etherscan_key = etherscan_key or ETHERSCAN_KEY
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "transactions.db"
        self.chain = str(chain).strip().lower()
        if self.chain not in CHAIN_ID_MAP:
            raise ValueError("Unsupported chain '{}'. Supported: {}".format(chain, sorted(set(CHAIN_ID_MAP.keys()))))
        self.chain_id = int(CHAIN_ID_MAP[self.chain])
        self.offline = bool(offline)

        has_alchemy = bool(ALCHEMY_KEY) or ("alchemy" in self.rpc_url.lower())
        self.requests_per_second = 25 if has_alchemy else 5
        self._min_interval = 1.0 / float(self.requests_per_second)
        self._last_request_ts = 0.0

        self._init_db()

    # ------------------------------
    # Public API
    # ------------------------------
    def fetch_transactions(
        self,
        addresses: Iterable[str],
        max_blocks_back: int = 200_000,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch transactions for addresses.

        Returns a DataFrame with columns:
        address, tx_hash, block_number, timestamp, from_addr, to_addr,
        value_wei, gas_price, gas_used, input_data_prefix
        """

        if isinstance(addresses, str):
            normalized = [addresses.strip().lower()] if addresses.strip() else []
        else:
            normalized = [str(a).lower() for a in addresses if str(a).strip()]
        if not normalized:
            return pd.DataFrame(columns=TX_COLUMNS)

        force_live_refresh = (not self.offline) and bool(self.etherscan_key)
        if force_live_refresh:
            # In live mode, refresh from explorer and avoid mixing stale synthetic
            # cache rows with live data for the same addresses.
            cached = pd.DataFrame(columns=TX_COLUMNS)
            cached_addresses = set()
            missing = list(normalized)
        else:
            cached = self._read_cached_transactions(normalized, max_blocks_back=max_blocks_back)
            cached_addresses = set(cached["address"].str.lower().unique()) if not cached.empty else set()
            missing = [a for a in normalized if a not in cached_addresses]

        fetched_parts = []
        if missing and not self.offline:
            if not self.etherscan_key:
                raise RuntimeError(
                    "Missing ETHERSCAN_API_KEY for live fetch. Set ETHERSCAN_API_KEY or pass offline=True/--offline."
                )
            for address in missing:
                df = self._fetch_transactions_from_etherscan(
                    address=address, max_blocks_back=max_blocks_back, limit=limit
                )
                if not df.empty:
                    fetched_parts.append(df)

        combined = [cached] + fetched_parts
        merged = pd.concat([df for df in combined if not df.empty], ignore_index=True) if any(
            not df.empty for df in combined
        ) else pd.DataFrame(columns=TX_COLUMNS)

        missing_after_online = set(missing) - set(merged["address"].str.lower().unique())
        if self.offline and (merged.empty or missing_after_online):
            fallback_addresses = list(set(missing) - set(merged["address"].str.lower().unique()))
            if not merged.empty:
                fallback_addresses = fallback_addresses or missing
            else:
                fallback_addresses = normalized

            fallback_df = self._synthetic_fallback_transactions(fallback_addresses)
            if not fallback_df.empty:
                merged = (
                    pd.concat([merged, fallback_df], ignore_index=True)
                    if not merged.empty
                    else fallback_df
                )
        elif missing_after_online:
            # Missing addresses in online mode are treated as a hard failure if they
            # have no cache and no fetched transactions.
            raise RuntimeError(
                "No live transactions returned for addresses {} on chain '{}' (chain_id={}). "
                "Use --offline for synthetic fallback.".format(sorted(missing_after_online), self.chain, self.chain_id)
            )

        if not merged.empty:
            merged = self._normalize_transactions_df(merged)
            self._write_cached_transactions(merged)
            if limit is not None and int(limit) > 0:
                merged = merged.sort_values(["timestamp", "block_number"]).tail(int(limit)).reset_index(drop=True)

        return merged

    def fetch_token_transfers(self, addresses: Iterable[str]) -> pd.DataFrame:
        """Fetch ERC-20 style transfers for addresses.

        Falls back to cache, then derives synthetic token transfer events from
        synthetic transaction fallback if needed.
        """

        if isinstance(addresses, str):
            normalized = [addresses.strip().lower()] if addresses.strip() else []
        else:
            normalized = [str(a).lower() for a in addresses if str(a).strip()]
        if not normalized:
            return pd.DataFrame(columns=TOKEN_TRANSFER_COLUMNS)

        force_live_refresh = (not self.offline) and bool(self.etherscan_key)
        if force_live_refresh:
            cached = pd.DataFrame(columns=TOKEN_TRANSFER_COLUMNS)
            cached_addresses = set()
            missing = list(normalized)
        else:
            cached = self._read_cached_token_transfers(normalized)
            cached_addresses = set(cached["address"].str.lower().unique()) if not cached.empty else set()
            missing = [a for a in normalized if a not in cached_addresses]

        online_parts = []
        if missing and not self.offline:
            if not self.etherscan_key:
                raise RuntimeError(
                    "Missing ETHERSCAN_API_KEY for live fetch. Set ETHERSCAN_API_KEY or pass offline=True/--offline."
                )
            for address in missing:
                df = self._fetch_token_transfers_from_etherscan(address)
                if not df.empty:
                    online_parts.append(df)

        merged = pd.concat([df for df in [cached] + online_parts if not df.empty], ignore_index=True) if (
            not cached.empty or any(not df.empty for df in online_parts)
        ) else pd.DataFrame(columns=TOKEN_TRANSFER_COLUMNS)

        uncovered = set(missing) - set(merged["address"].str.lower().unique()) if not merged.empty else set(missing)
        if uncovered and self.offline:
            synthetic_tx = self._synthetic_fallback_transactions(list(uncovered))
            if not synthetic_tx.empty:
                synthetic_token = pd.DataFrame(
                    {
                        "address": synthetic_tx["address"],
                        "tx_hash": synthetic_tx["tx_hash"],
                        "timestamp": synthetic_tx["timestamp"],
                        "from_addr": synthetic_tx["from_addr"],
                        "to_addr": synthetic_tx["to_addr"],
                        "token_symbol": "SYN",
                        "token_value": synthetic_tx["value_wei"].astype(float) / 1e18,
                    }
                )
                merged = (
                    pd.concat([merged, synthetic_token], ignore_index=True)
                    if not merged.empty
                    else synthetic_token
                )
        elif uncovered:
            raise RuntimeError(
                "No live token transfers returned for addresses {} on chain '{}' (chain_id={}). "
                "Use --offline for synthetic fallback.".format(sorted(uncovered), self.chain, self.chain_id)
            )

        if not merged.empty:
            merged = self._normalize_token_df(merged)
            self._write_cached_token_transfers(merged)

        return merged

    # ------------------------------
    # Online fetch helpers
    # ------------------------------
    def _fetch_transactions_from_etherscan(
        self, address: str, max_blocks_back: int, limit: Optional[int] = None
    ) -> pd.DataFrame:
        params = {
            "chainid": self.chain_id,
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",
            "apikey": self.etherscan_key,
        }
        if limit is not None and int(limit) > 0:
            params["page"] = 1
            params["offset"] = int(limit)
        payload = self._etherscan_request(params)
        rows = payload.get("result", []) if isinstance(payload, dict) else []
        if not rows:
            return pd.DataFrame(columns=TX_COLUMNS)

        tx_df = pd.DataFrame(rows)
        tx_df = tx_df.rename(
            columns={
                "hash": "tx_hash",
                "blockNumber": "block_number",
                "timeStamp": "timestamp",
                "from": "from_addr",
                "to": "to_addr",
                "value": "value_wei",
                "gasPrice": "gas_price",
                "gasUsed": "gas_used",
                "input": "input_data_prefix",
            }
        )
        tx_df["address"] = address.lower()
        tx_df = tx_df[[
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
        ]]

        tx_df["input_data_prefix"] = tx_df["input_data_prefix"].astype(str).str.slice(0, 10)
        tx_df = self._normalize_transactions_df(tx_df)
        tx_df = tx_df.sort_values(["timestamp", "block_number"]).reset_index(drop=True)

        if max_blocks_back and not tx_df.empty:
            latest = int(tx_df["block_number"].max())
            min_block = max(0, latest - int(max_blocks_back))
            tx_df = tx_df.loc[tx_df["block_number"] >= min_block].reset_index(drop=True)
        if limit is not None and int(limit) > 0 and not tx_df.empty:
            tx_df = tx_df.tail(int(limit)).reset_index(drop=True)

        return tx_df

    def _fetch_token_transfers_from_etherscan(self, address: str) -> pd.DataFrame:
        params = {
            "chainid": self.chain_id,
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "asc",
            "apikey": self.etherscan_key,
        }
        payload = self._etherscan_request(params)
        rows = payload.get("result", []) if isinstance(payload, dict) else []
        if not rows:
            return pd.DataFrame(columns=TOKEN_TRANSFER_COLUMNS)

        df = pd.DataFrame(rows)
        df = df.rename(
            columns={
                "hash": "tx_hash",
                "timeStamp": "timestamp",
                "from": "from_addr",
                "to": "to_addr",
                "tokenSymbol": "token_symbol",
                "value": "token_value",
            }
        )
        df["address"] = address.lower()
        keep = [
            "address",
            "tx_hash",
            "timestamp",
            "from_addr",
            "to_addr",
            "token_symbol",
            "token_value",
        ]
        df = df[keep]
        return self._normalize_token_df(df)

    def _etherscan_request(self, params: Dict[str, object]) -> Dict[str, object]:
        self._throttle()
        query = urllib.parse.urlencode(params)
        url = "https://api.etherscan.io/v2/api?{}".format(query)
        req = urllib.request.Request(url, headers={"User-Agent": "onchain-sybil-detector/0.2.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Etherscan V2 request failed for chain '{}' (chain_id={}): {}".format(self.chain, self.chain_id, exc)
            ) from exc

        status = str(payload.get("status", "1"))
        message = str(payload.get("message", ""))
        result = payload.get("result")
        no_tx = "No transactions found" in message or result == "No transactions found"
        if status == "0" and not no_tx:
            raise RuntimeError(
                "Etherscan V2 API error on chain '{}' (chain_id={}): message='{}' result='{}'".format(
                    self.chain, self.chain_id, message, result
                )
            )

        return payload

    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_ts = time.time()

    # ------------------------------
    # Offline fallback
    # ------------------------------
    def _synthetic_fallback_transactions(self, addresses: List[str]) -> pd.DataFrame:
        addresses = [a.lower() for a in addresses]
        synthetic = self._load_transactions_from_generator(addresses)
        if synthetic.empty:
            synthetic = self._minimal_synthetic_transactions(addresses)
        return self._normalize_transactions_df(synthetic)

    def _load_transactions_from_generator(self, addresses: List[str]) -> pd.DataFrame:
        try:
            from sybil_detector.datasets.synthetic_generator import generate_synthetic_sybil_network
        except Exception:
            generate_synthetic_sybil_network = None

        if generate_synthetic_sybil_network is not None:
            tx_df, _ = generate_synthetic_sybil_network(seed=17)
            if not tx_df.empty:
                tx_df["address"] = tx_df["address"].astype(str).str.lower()
                filtered = tx_df.loc[tx_df["address"].isin(addresses)].copy()
                missing = set(addresses) - set(filtered["address"].unique())
                if missing:
                    minimal = self._minimal_synthetic_transactions(list(missing))
                    filtered = pd.concat([filtered, minimal], ignore_index=True)
                return filtered[TX_COLUMNS]

        generator_path = Path(__file__).resolve().parents[2] / "datasets" / "synthetic_generator.py"
        if not generator_path.exists():
            return pd.DataFrame(columns=TX_COLUMNS)

        spec = importlib.util.spec_from_file_location("_sybil_synth", str(generator_path))
        if spec is None or spec.loader is None:
            return pd.DataFrame(columns=TX_COLUMNS)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        generate = getattr(module, "generate_synthetic_sybil_network", None)
        if generate is None:
            return pd.DataFrame(columns=TX_COLUMNS)

        tx_df, _ = generate(seed=17)
        if tx_df.empty:
            return pd.DataFrame(columns=TX_COLUMNS)

        tx_df["address"] = tx_df["address"].astype(str).str.lower()
        filtered = tx_df.loc[tx_df["address"].isin(addresses)].copy()

        # Create minimal rows for unknown addresses.
        missing = set(addresses) - set(filtered["address"].unique())
        if missing:
            minimal = self._minimal_synthetic_transactions(list(missing))
            filtered = pd.concat([filtered, minimal], ignore_index=True)

        return filtered[TX_COLUMNS]

    def _minimal_synthetic_transactions(self, addresses: List[str]) -> pd.DataFrame:
        now = int(time.time())
        rows = []
        for idx, address in enumerate(addresses):
            from_addr = "0x" + ("f" * 39) + str(idx % 10)
            tx_hash = "0x" + ("0" * 56) + "{:08x}".format(idx)
            rows.append(
                {
                    "address": address,
                    "tx_hash": tx_hash,
                    "block_number": 18_000_000 + idx,
                    "timestamp": now - idx * 300,
                    "from_addr": from_addr,
                    "to_addr": address,
                    "value_wei": int(1e17),
                    "gas_price": int(30e9),
                    "gas_used": 21_000,
                    "input_data_prefix": "0xa9059cbb",
                }
            )
            rows.append(
                {
                    "address": address,
                    "tx_hash": tx_hash + "1",
                    "block_number": 18_000_001 + idx,
                    "timestamp": now - idx * 120,
                    "from_addr": address,
                    "to_addr": "0x" + ("e" * 39) + str(idx % 10),
                    "value_wei": int(6e16),
                    "gas_price": int(31e9),
                    "gas_used": 52_000,
                    "input_data_prefix": "0x2e1a7d4d",
                }
            )

        return pd.DataFrame(rows, columns=TX_COLUMNS)

    # ------------------------------
    # Cache helpers
    # ------------------------------
    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    address TEXT NOT NULL,
                    tx_hash TEXT NOT NULL,
                    block_number INTEGER,
                    timestamp INTEGER,
                    from_addr TEXT,
                    to_addr TEXT,
                    value_wei INTEGER,
                    gas_price INTEGER,
                    gas_used INTEGER,
                    input_data_prefix TEXT,
                    PRIMARY KEY (address, tx_hash)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_transfers (
                    address TEXT NOT NULL,
                    tx_hash TEXT NOT NULL,
                    timestamp INTEGER,
                    from_addr TEXT,
                    to_addr TEXT,
                    token_symbol TEXT,
                    token_value REAL,
                    PRIMARY KEY (address, tx_hash)
                )
                """
            )
            conn.commit()

    def _read_cached_transactions(
        self, addresses: List[str], max_blocks_back: int
    ) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame(columns=TX_COLUMNS)

        placeholders = ",".join(["?"] * len(addresses))
        query = (
            "SELECT address, tx_hash, block_number, timestamp, from_addr, to_addr, "
            "value_wei, gas_price, gas_used, input_data_prefix "
            "FROM transactions WHERE lower(address) IN ({})"
        ).format(placeholders)

        with sqlite3.connect(str(self.db_path)) as conn:
            df = pd.read_sql_query(query, conn, params=[a.lower() for a in addresses])

        if df.empty:
            return df

        df = self._normalize_transactions_df(df)
        if max_blocks_back and not df.empty:
            latest = int(df["block_number"].max())
            min_block = max(0, latest - int(max_blocks_back))
            df = df.loc[df["block_number"] >= min_block].reset_index(drop=True)
        return df

    def _write_cached_transactions(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        normalized = self._normalize_transactions_df(df)
        records = [tuple(row[c] for c in TX_COLUMNS) for _, row in normalized.iterrows()]

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO transactions
                (address, tx_hash, block_number, timestamp, from_addr, to_addr,
                 value_wei, gas_price, gas_used, input_data_prefix)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def _read_cached_token_transfers(self, addresses: List[str]) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame(columns=TOKEN_TRANSFER_COLUMNS)

        placeholders = ",".join(["?"] * len(addresses))
        query = (
            "SELECT address, tx_hash, timestamp, from_addr, to_addr, token_symbol, token_value "
            "FROM token_transfers WHERE lower(address) IN ({})"
        ).format(placeholders)
        with sqlite3.connect(str(self.db_path)) as conn:
            df = pd.read_sql_query(query, conn, params=[a.lower() for a in addresses])
        if df.empty:
            return df
        return self._normalize_token_df(df)

    def _write_cached_token_transfers(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        normalized = self._normalize_token_df(df)
        records = [tuple(row[c] for c in TOKEN_TRANSFER_COLUMNS) for _, row in normalized.iterrows()]
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO token_transfers
                (address, tx_hash, timestamp, from_addr, to_addr, token_symbol, token_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    # ------------------------------
    # Data normalization
    # ------------------------------
    def _normalize_transactions_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in TX_COLUMNS:
            if col not in out.columns:
                out[col] = 0 if col not in {"address", "tx_hash", "from_addr", "to_addr", "input_data_prefix"} else ""

        out["address"] = out["address"].astype(str).str.lower()
        out["tx_hash"] = out["tx_hash"].astype(str)
        out["from_addr"] = out["from_addr"].astype(str).str.lower()
        out["to_addr"] = out["to_addr"].astype(str).str.lower()
        out["input_data_prefix"] = out["input_data_prefix"].astype(str).str.slice(0, 10)

        for col in ["block_number", "timestamp", "value_wei", "gas_price", "gas_used"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)

        out = out[TX_COLUMNS].drop_duplicates(subset=["address", "tx_hash"]).sort_values(
            ["address", "timestamp", "block_number"]
        )
        return out.reset_index(drop=True)

    def _normalize_token_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in TOKEN_TRANSFER_COLUMNS:
            if col not in out.columns:
                out[col] = 0 if col in {"timestamp", "token_value"} else ""

        out["address"] = out["address"].astype(str).str.lower()
        out["tx_hash"] = out["tx_hash"].astype(str)
        out["from_addr"] = out["from_addr"].astype(str).str.lower()
        out["to_addr"] = out["to_addr"].astype(str).str.lower()
        out["token_symbol"] = out["token_symbol"].fillna("UNK").astype(str)
        out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0).astype(np.int64)
        out["token_value"] = pd.to_numeric(out["token_value"], errors="coerce").fillna(0.0)

        out = out[TOKEN_TRANSFER_COLUMNS].drop_duplicates(subset=["address", "tx_hash"])
        return out.reset_index(drop=True)


__all__ = [
    "ChainDataFetcher",
    "ETHERSCAN_KEY",
    "ALCHEMY_KEY",
    "RPC_URL",
    "TX_COLUMNS",
    "TOKEN_TRANSFER_COLUMNS",
]
