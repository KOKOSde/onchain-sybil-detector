"""CLI for onchain-sybil-detector OSD modes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from sybil_detector import SybilDetector, extract_features
from sybil_detector.data_ingestion import ChainDataFetcher
from sybil_detector.visualization import generate_report

TX_REQUIRED_COLUMNS = {
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
}

CHAIN_ALIASES = {
    "eth": "ethereum",
    "ethereum": "ethereum",
    "base": "base",
    "bnb": "bnb",
    "bsc": "bnb",
    "arb": "arbitrum",
    "arbitrum": "arbitrum",
    "op": "optimism",
    "optimism": "optimism",
    "polygon": "polygon",
    "matic": "polygon",
}


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default_config_osd.yaml"


def _load_adversarial_module():
    try:
        from datasets import adversarial_simulator_osd as adv_mod  # type: ignore
        return adv_mod
    except Exception:
        root = Path(__file__).resolve().parents[2]
        module_path = root / "datasets" / "adversarial_simulator_osd.py"
        spec = importlib.util.spec_from_file_location("osd_adversarial_simulator", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load adversarial simulator from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


def _expand_env_vars(value):
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            return os.getenv(key, "")

        return pattern.sub(repl, value)
    return value


def load_config(config_path: Optional[str]) -> Dict[str, object]:
    path = Path(config_path) if config_path else _default_config_path()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _expand_env_vars(raw)


def _normalize_chain(chain: str) -> str:
    return CHAIN_ALIASES.get(str(chain).strip().lower(), "ethereum")


def _resolve_chain_rpc(chain: str, config: Dict[str, object]) -> str:
    chain = _normalize_chain(chain)
    alchemy_key = os.getenv("ALCHEMY_API_KEY") or str(config.get("alchemy_api_key", "")).strip()
    if chain == "ethereum":
        if alchemy_key:
            return f"https://eth-mainnet.g.alchemy.com/v2/{alchemy_key}"
        return "https://rpc.ankr.com/eth"
    if chain == "base":
        if alchemy_key:
            return f"https://base-mainnet.g.alchemy.com/v2/{alchemy_key}"
        return "https://mainnet.base.org"
    if chain == "bnb":
        return "https://bsc-dataseed.binance.org"
    if chain == "arbitrum":
        return "https://arb1.arbitrum.io/rpc"
    if chain == "optimism":
        return "https://mainnet.optimism.io"
    if chain == "polygon":
        return "https://polygon-rpc.com"
    return "https://rpc.ankr.com/eth"


def _load_addresses_or_transactions(path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if TX_REQUIRED_COLUMNS.issubset(set(df.columns)):
        return df, []

    if "address" in df.columns:
        addresses = df["address"].dropna().astype(str).str.lower().tolist()
    else:
        first_col = df.columns[0]
        addresses = df[first_col].dropna().astype(str).str.lower().tolist()
    return pd.DataFrame(), sorted(set(addresses))


def _build_fetcher(config: Dict[str, object], chain: str) -> ChainDataFetcher:
    etherscan_key = os.getenv("ETHERSCAN_API_KEY") or str(config.get("etherscan_api_key", "")).strip() or None
    rpc = _resolve_chain_rpc(chain, config)
    return ChainDataFetcher(rpc_url=rpc, etherscan_key=etherscan_key, cache_dir="cache")


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copyfile(src, dst)


def cmd_simulate(args: argparse.Namespace) -> int:
    adv_mod = _load_adversarial_module()
    levels = None
    if args.levels:
        levels = [int(x.strip()) for x in str(args.levels).split(",") if x.strip()]

    tx, labels = adv_mod.generate_adversarial_sybils(
        difficulty=int(args.difficulty),
        num_clusters=int(args.num_clusters),
        wallets_per_cluster=int(args.wallets_per_cluster),
        num_legit=int(args.num_legit),
        seed=int(args.seed),
        enabled_levels=levels,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tx.to_csv(out, index=False)

    labels_out = out.with_name(out.stem + "_labels.csv")
    labels.to_csv(labels_out, index=False)

    print(json.dumps({"transactions": str(out), "labels": str(labels_out), "rows": int(len(tx))}))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    chain = _normalize_chain(args.chain)

    tx_df, addresses = _load_addresses_or_transactions(args.addresses)
    if tx_df.empty:
        fetcher = _build_fetcher(config, chain)
        tx_df = fetcher.fetch_transactions(addresses, max_blocks_back=int(args.max_blocks_back))

    features = extract_features(tx_df)
    min_cluster_size = int((config.get("detection", {}) or {}).get("min_cluster_size", 3))
    detector = SybilDetector(min_cluster_size=min_cluster_size, min_samples=2)
    clusters = detector.fit_predict(features)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    report_written = False
    try:
        from sybil_detector.report_osd import generate_analyst_reports

        generated = generate_analyst_reports(tx_df, clusters, output_dir=str(out.parent))
        ext = out.suffix.lower().lstrip(".")
        if ext == "json":
            _safe_copy(Path(generated["json"]), out)
        elif ext in {"md", "markdown"}:
            _safe_copy(Path(generated["markdown"]), out)
        else:
            _safe_copy(Path(generated["html"]), out)
        report_written = True
    except Exception:
        report_written = False

    if not report_written:
        report = generate_report(clusters, format="html" if out.suffix.lower() == ".html" else "text")
        out.write_text(report, encoding="utf-8")

    print(json.dumps({"output": str(out), "clusters": int((clusters["cluster_id"] != -1).sum())}))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    chain = _normalize_chain(args.chain)
    wallets = [w.strip().lower() for w in str(args.wallets).split(",") if w.strip()]
    if len(wallets) < 2:
        raise ValueError("--wallets must contain at least two addresses")

    if args.transactions:
        tx_df = pd.read_csv(args.transactions)
    else:
        fetcher = _build_fetcher(config, chain)
        tx_df = fetcher.fetch_transactions(wallets, max_blocks_back=int(args.max_blocks_back))

    try:
        from sybil_detector.explainer_osd import explain_wallet_linkage

        result = explain_wallet_linkage(wallets=wallets, transactions=tx_df)
    except Exception:
        features = extract_features(tx_df)
        subset = features[features["address"].isin(wallets)]
        shared_funder = subset["common_funder_address"].replace("", pd.NA).dropna()
        same_funder = bool(not shared_funder.empty and shared_funder.nunique() == 1)
        gas_span = float(subset["median_gas_price"].max() - subset["median_gas_price"].min())
        gas_median = float(subset["median_gas_price"].median()) if not subset.empty else 0.0
        gas_rel = gas_span / gas_median if gas_median > 0 else 1.0
        evidence = []
        if same_funder:
            evidence.append(f"Funded from same upstream wallet ({shared_funder.iloc[0]})")
        evidence.append(f"Near-identical gas strategy delta={gas_span:.0f} (relative {gas_rel:.3f})")
        confidence = float(max(0.0, min(1.0, 0.45 + (0.35 if same_funder else 0.0) + (0.2 if gas_rel < 0.05 else 0.0))))
        result = {
            "wallets": wallets,
            "evidence_sentences": evidence,
            "confidence": confidence,
            "linkage_strength": confidence,
        }

    print(json.dumps(result, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    adv_mod = _load_adversarial_module()
    results = adv_mod.run_adversarial_detection_benchmark(
        detector_factory=lambda: SybilDetector(min_cluster_size=3, min_samples=2),
        levels=range(1, 9),
        num_clusters=int(args.num_clusters),
        wallets_per_cluster=int(args.wallets_per_cluster),
        num_legit=int(args.num_legit),
        seed=int(args.seed),
    )

    rows = []
    for level, metrics in results.items():
        row = {"level": level}
        row.update(metrics)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(json.dumps({"output": str(out), "levels": len(rows)}))
    return 0


def cmd_airdrop_scan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    chain = _normalize_chain(args.chain)

    if args.addresses:
        tx_df, addresses = _load_addresses_or_transactions(args.addresses)
        if tx_df.empty:
            fetcher = _build_fetcher(config, chain)
            tx_df = fetcher.fetch_transactions(addresses, max_blocks_back=int(args.max_blocks_back))
    else:
        adv_mod = _load_adversarial_module()
        tx_df, _ = adv_mod.generate_adversarial_sybils(difficulty=2, num_clusters=6, wallets_per_cluster=10, seed=42)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    result_obj: Dict[str, object]
    try:
        from sybil_detector.airdrop_hunter_osd import scan_airdrop_campaign

        result_obj = scan_airdrop_campaign(
            transactions=tx_df,
            campaign_contract=str(args.contract).lower(),
            chain=chain,
        )
    except Exception:
        features = extract_features(tx_df)
        clusters = SybilDetector(min_cluster_size=3, min_samples=2).fit_predict(features)
        flagged = clusters[clusters["sybil_probability"] >= 0.7]
        abuse_pct = float(100.0 * len(flagged) / max(1, len(clusters)))
        result_obj = {
            "campaign_contract": str(args.contract).lower(),
            "chain": chain,
            "likely_farmer_clusters": sorted(flagged["cluster_id"].unique().tolist()),
            "estimated_sybil_participant_pct": abuse_pct,
        }

    if out.suffix.lower() == ".html":
        html = [
            "<html><body>",
            "<h1>Airdrop Hunter Report</h1>",
            f"<p>Contract: {result_obj.get('campaign_contract', str(args.contract).lower())}</p>",
            f"<p>Chain: {result_obj.get('chain', chain)}</p>",
            f"<pre>{json.dumps(result_obj, indent=2)}</pre>",
            "</body></html>",
        ]
        out.write_text("\n".join(html), encoding="utf-8")
    elif out.suffix.lower() == ".csv":
        pd.DataFrame([result_obj]).to_csv(out, index=False)
    else:
        out.write_text(json.dumps(result_obj, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(out)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sybil-detector", description="OSD CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Detect sybil clusters from address list or tx CSV")
    scan.add_argument("--addresses", required=True, help="CSV of addresses or transactions")
    scan.add_argument("--chain", default="eth")
    scan.add_argument("--out", required=True)
    scan.add_argument("--config", default=None)
    scan.add_argument("--max-blocks-back", default=200000, type=int)
    scan.set_defaults(func=cmd_scan)

    sim = sub.add_parser("simulate", help="Generate adversarial synthetic dataset")
    sim.add_argument("--difficulty", required=True, type=int)
    sim.add_argument("--num-clusters", required=True, type=int)
    sim.add_argument("--wallets-per-cluster", required=True, type=int)
    sim.add_argument("--num-legit", default=300, type=int)
    sim.add_argument("--seed", default=42, type=int)
    sim.add_argument("--levels", default=None, help="Optional comma-separated level toggles")
    sim.add_argument("--out", required=True)
    sim.set_defaults(func=cmd_simulate)

    explain = sub.add_parser("explain", help="Explain why wallets are linked")
    explain.add_argument("--wallets", required=True, help="Comma-separated wallet addresses")
    explain.add_argument("--chain", default="eth")
    explain.add_argument("--transactions", default=None, help="Optional transaction CSV")
    explain.add_argument("--config", default=None)
    explain.add_argument("--max-blocks-back", default=200000, type=int)
    explain.set_defaults(func=cmd_explain)

    bench = sub.add_parser("benchmark", help="Run adversarial benchmark 1..8")
    bench.add_argument("--out", required=True)
    bench.add_argument("--seed", default=42, type=int)
    bench.add_argument("--num-clusters", default=10, type=int)
    bench.add_argument("--wallets-per-cluster", default=12, type=int)
    bench.add_argument("--num-legit", default=300, type=int)
    bench.set_defaults(func=cmd_benchmark)

    airdrop = sub.add_parser("airdrop-scan", help="Campaign-specific sybil scan")
    airdrop.add_argument("--contract", required=True)
    airdrop.add_argument("--chain", default="eth")
    airdrop.add_argument("--addresses", default=None, help="Optional participant CSV")
    airdrop.add_argument("--config", default=None)
    airdrop.add_argument("--max-blocks-back", default=200000, type=int)
    airdrop.add_argument("--out", required=True)
    airdrop.set_defaults(func=cmd_airdrop_scan)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
