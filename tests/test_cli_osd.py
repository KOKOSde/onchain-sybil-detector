import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


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


def test_cli_simulate_scan_explain_benchmark_smoke(tmp_path: Path) -> None:
    root = _repo_root()
    tx_csv = tmp_path / "synthetic.csv"
    sim = _run_cli(
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
    assert sim.returncode == 0, sim.stderr
    assert tx_csv.exists()
    labels_csv = tx_csv.with_name(tx_csv.stem + "_labels.csv")
    assert labels_csv.exists()

    report_html = tmp_path / "report.html"
    scan = _run_cli(
        [
            "scan",
            "--addresses",
            str(tx_csv),
            "--chain",
            "eth",
            "--out",
            str(report_html),
        ],
        cwd=root,
    )
    assert scan.returncode == 0, scan.stderr
    assert report_html.exists()

    labels_df = pd.read_csv(labels_csv)
    wallets = labels_df["address"].head(3).tolist()
    explain = _run_cli(
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
    assert explain.returncode == 0, explain.stderr
    parsed = json.loads(explain.stdout)
    assert "confidence_score" in parsed
    assert len(parsed["wallets"]) == 3

    benchmark_csv = tmp_path / "benchmark.csv"
    bench = _run_cli(
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
    assert bench.returncode == 0, bench.stderr
    assert benchmark_csv.exists()
    bench_df = pd.read_csv(benchmark_csv)
    assert len(bench_df) == 8
