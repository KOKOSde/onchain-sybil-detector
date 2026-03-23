import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sybil_detector.datasets.synthetic_generator import generate_synthetic_sybil_network


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cli_env() -> dict:
    root = _repo_root()
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root / 'src'}:{root}:{existing}" if existing else f"{root / 'src'}:{root}"
    return env


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "sybil_detector.cli_osd", *args]
    return subprocess.run(cmd, cwd=str(cwd), env=_cli_env(), capture_output=True, text=True, check=False)


@pytest.fixture
def prepared_synthetic_csv(tmp_path: Path) -> dict:
    tx, labels = generate_synthetic_sybil_network(
        num_legit=80,
        num_sybil_clusters=4,
        addrs_per_cluster=8,
        seed=123,
    )
    tx_csv = tmp_path / "synthetic_tx.csv"
    labels_csv = tmp_path / "synthetic_labels.csv"
    tx.to_csv(tx_csv, index=False)
    labels.to_csv(labels_csv, index=False)
    return {
        "root": _repo_root(),
        "tx_csv": tx_csv,
        "labels_csv": labels_csv,
    }


def test_cli_help_root() -> None:
    result = _run_cli(["--help"], cwd=_repo_root())
    assert result.returncode == 0
    assert "sybil-detector" in result.stdout


@pytest.mark.parametrize("subcommand", ["scan", "simulate", "explain", "benchmark", "airdrop-scan"])
def test_cli_help_subcommands(subcommand: str) -> None:
    result = _run_cli([subcommand, "--help"], cwd=_repo_root())
    assert result.returncode == 0
    assert subcommand in result.stdout


def test_cli_simulate_command(tmp_path: Path) -> None:
    root = _repo_root()
    tx_csv = tmp_path / "simulated.csv"
    result = _run_cli(
        [
            "simulate",
            "--difficulty",
            "1",
            "--num-clusters",
            "3",
            "--wallets-per-cluster",
            "6",
            "--num-legit",
            "40",
            "--out",
            str(tx_csv),
        ],
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    assert tx_csv.exists()
    assert tx_csv.with_name(tx_csv.stem + "_labels.csv").exists()

    payload = json.loads(result.stdout)
    assert payload["rows"] > 0


def test_cli_scan_command(prepared_synthetic_csv: dict, tmp_path: Path) -> None:
    root = prepared_synthetic_csv["root"]
    tx_csv = prepared_synthetic_csv["tx_csv"]
    report_html = tmp_path / "scan_report.html"

    result = _run_cli(
        ["scan", "--addresses", str(tx_csv), "--chain", "eth", "--out", str(report_html)],
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    assert report_html.exists()
    payload = json.loads(result.stdout)
    assert payload["output"].endswith("scan_report.html")


def test_cli_simulate_then_scan_finds_clusters(tmp_path: Path) -> None:
    root = _repo_root()
    tx_csv = tmp_path / "pipeline_synthetic.csv"
    report_html = tmp_path / "pipeline_report.html"

    simulate = _run_cli(
        [
            "simulate",
            "--difficulty",
            "1",
            "--num-clusters",
            "5",
            "--wallets-per-cluster",
            "10",
            "--num-legit",
            "120",
            "--seed",
            "42",
            "--out",
            str(tx_csv),
        ],
        cwd=root,
    )
    assert simulate.returncode == 0, simulate.stderr
    assert tx_csv.exists()

    scan = _run_cli(
        ["scan", "--addresses", str(tx_csv), "--chain", "eth", "--out", str(report_html)],
        cwd=root,
    )
    assert scan.returncode == 0, scan.stderr
    payload = json.loads(scan.stdout)
    assert int(payload["clusters"]) > 0

    html = report_html.read_text(encoding="utf-8")
    match = re.search(r"Clusters analyzed:\s*(\d+)", html)
    assert match is not None
    assert int(match.group(1)) > 0


def test_cli_explain_command(prepared_synthetic_csv: dict) -> None:
    root = prepared_synthetic_csv["root"]
    tx_csv = prepared_synthetic_csv["tx_csv"]
    labels_df = pd.read_csv(prepared_synthetic_csv["labels_csv"])
    wallets = labels_df["address"].head(3).tolist()

    result = _run_cli(
        [
            "explain",
            "--wallets",
            ",".join(wallets),
            "--chain",
            "eth",
            "--transactions",
            str(tx_csv),
        ],
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert "confidence_score" in parsed
    assert len(parsed["wallets"]) == 3


def test_cli_benchmark_command(tmp_path: Path) -> None:
    root = _repo_root()
    benchmark_csv = tmp_path / "benchmark.csv"
    result = _run_cli(
        [
            "benchmark",
            "--out",
            str(benchmark_csv),
            "--num-clusters",
            "2",
            "--wallets-per-cluster",
            "5",
            "--num-legit",
            "30",
        ],
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    assert benchmark_csv.exists()
    bench_df = pd.read_csv(benchmark_csv)
    assert len(bench_df) == 8


def test_cli_airdrop_scan_command(prepared_synthetic_csv: dict, tmp_path: Path) -> None:
    root = prepared_synthetic_csv["root"]
    tx_csv = prepared_synthetic_csv["tx_csv"]
    tx_df = pd.read_csv(tx_csv)
    contract = str(tx_df["to_addr"].iloc[0])
    out_html = tmp_path / "airdrop_report.html"

    result = _run_cli(
        [
            "airdrop-scan",
            "--contract",
            contract,
            "--chain",
            "base",
            "--addresses",
            str(tx_csv),
            "--out",
            str(out_html),
        ],
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    assert out_html.exists()
    assert "Airdrop Hunter Report" in out_html.read_text(encoding="utf-8")


def test_cli_invalid_subcommand_returns_error() -> None:
    result = _run_cli(["unknown-cmd"], cwd=_repo_root())
    assert result.returncode != 0
    assert "usage" in result.stderr.lower()


def test_cli_invalid_explain_wallets_returns_clean_error(prepared_synthetic_csv: dict) -> None:
    root = prepared_synthetic_csv["root"]
    tx_csv = prepared_synthetic_csv["tx_csv"]

    result = _run_cli(
        ["explain", "--wallets", "0xabc", "--chain", "eth", "--transactions", str(tx_csv)],
        cwd=root,
    )
    assert result.returncode != 0
    err = (result.stderr or result.stdout).lower()
    assert "at least two" in err
