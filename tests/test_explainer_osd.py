import pandas as pd

from sybil_detector.explainer_osd import explain_wallet_linkage


def _linked_transactions() -> pd.DataFrame:
    rows = []

    def add_tx(tx_hash: str, ts: int, from_addr: str, to_addr: str, gas_price: int, method: str) -> None:
        base = {
            "tx_hash": tx_hash,
            "block_number": 19_000_000 + (ts - 1_700_000_000) // 12,
            "timestamp": ts,
            "from_addr": from_addr,
            "to_addr": to_addr,
            "value_wei": 10**16,
            "gas_price": gas_price,
            "gas_used": 21_000,
            "input_data_prefix": method,
        }
        rows.append({"address": from_addr, **base})
        rows.append({"address": to_addr, **base})

    funder = "0x00000000000000000000000000000000000f0001"
    a = "0x00000000000000000000000000000000000a0001"
    b = "0x00000000000000000000000000000000000a0002"

    add_tx("0x1", 1_700_000_000, funder, a, 20_000_000_000, "0x")
    add_tx("0x2", 1_700_000_120, funder, b, 20_000_000_000, "0x")

    # Shared minute-bucket behavior and similar method sequence.
    add_tx("0x3", 1_700_010_000, a, "0x00000000000000000000000000000000000c0001", 20_020_000_000, "0xa9059cbb")
    add_tx("0x4", 1_700_010_030, b, "0x00000000000000000000000000000000000c0002", 20_015_000_000, "0xa9059cbb")
    add_tx("0x5", 1_700_020_000, a, "0x00000000000000000000000000000000000c0003", 20_010_000_000, "0x23b872dd")
    add_tx("0x6", 1_700_020_020, b, "0x00000000000000000000000000000000000c0004", 20_008_000_000, "0x23b872dd")
    add_tx("0x7", 1_700_030_000, a, "0x00000000000000000000000000000000000c0005", 20_005_000_000, "0x095ea7b3")
    add_tx("0x8", 1_700_030_040, b, "0x00000000000000000000000000000000000c0006", 20_000_000_000, "0x095ea7b3")

    return pd.DataFrame(rows)


def test_explainer_returns_plain_english_linkage_and_confidence() -> None:
    tx = _linked_transactions()
    wallets = [
        "0x00000000000000000000000000000000000a0001",
        "0x00000000000000000000000000000000000a0002",
    ]

    chains_data = pd.DataFrame(
        [
            {"address": wallets[0], "chain": "ethereum", "timestamp": 1_700_040_000},
            {"address": wallets[0], "chain": "base", "timestamp": 1_700_040_120},
            {"address": wallets[1], "chain": "ethereum", "timestamp": 1_700_040_010},
            {"address": wallets[1], "chain": "base", "timestamp": 1_700_040_170},
        ]
    )

    result = explain_wallet_linkage(wallets=wallets, transactions=tx, chains_data=chains_data)

    assert result["wallets"] == wallets
    assert result["confidence_score"] > 0
    assert result["linkage_strength"] in {"weak", "moderate", "strong"}
    assert len(result["evidence_sentences"]) >= 1
    assert any("Funded from same upstream wallet" in s for s in result["evidence_sentences"])
