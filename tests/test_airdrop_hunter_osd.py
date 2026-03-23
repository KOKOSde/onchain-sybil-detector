from sybil_detector.airdrop_hunter_osd import (
    detect_campaign_participants,
    run_airdrop_hunter,
    scan_airdrop_campaign,
)
from sybil_detector.datasets.synthetic_generator import generate_synthetic_sybil_network


def test_airdrop_hunter_outputs_wallet_scores_and_campaign_summary() -> None:
    tx, labels = generate_synthetic_sybil_network(
        num_legit=160,
        num_sybil_clusters=6,
        addrs_per_cluster=14,
        seed=21,
    )

    sybil_participants = labels.loc[labels["is_sybil"] == 1, "address"].head(60).tolist()
    legit_participants = labels.loc[labels["is_sybil"] == 0, "address"].head(30).tolist()
    participants = sybil_participants + legit_participants

    result = run_airdrop_hunter(
        participant_addresses=participants,
        transactions=tx,
        chain="eth",
        min_cluster_size=3,
        min_samples=2,
        confidence_threshold=0.55,
    )

    assert result["chain"] == "ethereum"
    assert result["participant_count"] == len(participants)
    assert len(result["wallet_scores"]) > 0
    assert set(result["campaign_summary"].keys()) >= {
        "estimated_sybil_participants",
        "estimated_sybil_percentage",
        "confidence_threshold",
        "cluster_count",
    }

    for row in result["wallet_scores"]:
        score = float(row["wallet_suspicion_score"])
        assert 0.0 <= score <= 1.0


def test_scan_airdrop_campaign_wrapper_sets_contract_and_detects_participants() -> None:
    tx, labels = generate_synthetic_sybil_network(
        num_legit=80,
        num_sybil_clusters=4,
        addrs_per_cluster=10,
        seed=11,
    )
    participants = labels["address"].head(40).tolist()

    campaign_contract = str(tx["to_addr"].iloc[0]).lower()
    discovered = detect_campaign_participants(tx, campaign_contract)
    assert isinstance(discovered, list)

    result = scan_airdrop_campaign(
        transactions=tx,
        campaign_contract=campaign_contract,
        chain="base",
        participant_addresses=participants,
    )
    assert result["campaign_contract"] == campaign_contract
    assert result["chain"] == "base"
