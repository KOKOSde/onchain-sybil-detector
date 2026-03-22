import pandas as pd

from sybil_detector.chains_osd import (
    attach_chain_defaults,
    detect_cross_chain_coordination,
    get_chain_config,
    list_supported_chains,
    normalize_chain_name,
)


def test_chain_registry_and_aliases() -> None:
    chains = list_supported_chains()
    assert chains == ["arbitrum", "base", "bnb", "ethereum", "optimism", "polygon"]

    assert normalize_chain_name("eth") == "ethereum"
    assert normalize_chain_name("bsc") == "bnb"
    assert normalize_chain_name("op") == "optimism"

    cfg = get_chain_config("arb")
    assert cfg.name == "arbitrum"
    assert cfg.block_time_sec > 0


def test_attach_chain_defaults_and_cross_chain_coordination() -> None:
    tx = pd.DataFrame(
        [
            {
                "address": "0xaaa",
                "tx_hash": "0x1",
                "block_number": 1,
                "timestamp": 1_700_000_000,
                "from_addr": "0xfunder",
                "to_addr": "0xaaa",
                "value_wei": 100,
                "gas_price": 20_000_000_000,
                "gas_used": 21_000,
                "input_data_prefix": "0x",
                "chain": "ethereum",
            },
            {
                "address": "0xaaa",
                "tx_hash": "0x2",
                "block_number": 2,
                "timestamp": 1_700_000_120,
                "from_addr": "0xfunder",
                "to_addr": "0xaaa",
                "value_wei": 100,
                "gas_price": 20_100_000_000,
                "gas_used": 21_000,
                "input_data_prefix": "0x",
                "chain": "base",
            },
            {
                "address": "0xbbb",
                "tx_hash": "0x3",
                "block_number": 3,
                "timestamp": 1_700_000_180,
                "from_addr": "0xfunder",
                "to_addr": "0xbbb",
                "value_wei": 100,
                "gas_price": 20_050_000_000,
                "gas_used": 21_000,
                "input_data_prefix": "0x",
                "chain": "ethereum",
            },
            {
                "address": "0xbbb",
                "tx_hash": "0x4",
                "block_number": 4,
                "timestamp": 1_700_000_260,
                "from_addr": "0xfunder",
                "to_addr": "0xbbb",
                "value_wei": 100,
                "gas_price": 20_040_000_000,
                "gas_used": 21_000,
                "input_data_prefix": "0x",
                "chain": "base",
            },
        ]
    )

    attached = attach_chain_defaults(tx.drop(columns=["chain"]), "polygon")
    assert (attached["chain"] == "polygon").all()
    assert "chain_block_time_sec" in attached.columns

    metrics = detect_cross_chain_coordination(tx)
    assert metrics["multi_chain_wallet_count"] >= 2
    assert metrics["coordination_score"] > 0
    assert isinstance(metrics["evidence"], list)
