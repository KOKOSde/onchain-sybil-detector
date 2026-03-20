"""Clustering and coordination scoring for Sybil detection."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import RobustScaler

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None


class SybilDetector(object):
    """Unsupervised Sybil detector using HDBSCAN + coordination scoring."""

    def __init__(self, min_cluster_size: int = 3, min_samples: int = 2) -> None:
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples)
        self.cluster_evidence_: Dict[int, Dict[str, float]] = {}
        self.cluster_members_: Dict[int, List[str]] = {}
        self.results_: pd.DataFrame = pd.DataFrame()

    def fit_predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Cluster addresses and return Sybil probabilities with explanations."""

        output_columns = [
            "address",
            "cluster_id",
            "sybil_probability",
            "top_3_features",
            "coordination_breakdown",
        ]

        if features.empty:
            self.results_ = pd.DataFrame(columns=output_columns)
            return self.results_

        if "address" not in features.columns:
            raise ValueError("features must include an 'address' column")

        df = features.copy().reset_index(drop=True)
        df["address"] = df["address"].astype(str).str.lower()

        if "common_funder_address" not in df.columns:
            df["common_funder_address"] = ""
        df["common_funder_address"] = (
            df["common_funder_address"].fillna("").astype(str).str.lower()
        )

        numeric_cols = [
            c
            for c in df.columns
            if c not in {"address", "common_funder_address", "hour_histogram"}
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            raise ValueError("features must include numeric columns for clustering")

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        scaler = RobustScaler()
        scaled = scaler.fit_transform(df[numeric_cols].values)

        if hdbscan is not None:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
            )
            base_labels = clusterer.fit_predict(scaled)
        else:  # pragma: no cover
            clusterer = DBSCAN(eps=1.15, min_samples=self.min_samples)
            base_labels = clusterer.fit_predict(scaled)

        labels = self._refine_labels(df, np.asarray(base_labels, dtype=int))
        df["cluster_id"] = labels

        self.cluster_evidence_.clear()
        self.cluster_members_.clear()

        for cluster_id in sorted([c for c in df["cluster_id"].unique() if c != -1]):
            members = df.loc[df["cluster_id"] == cluster_id].copy()
            evidence = self._coordination_breakdown(members)
            self.cluster_evidence_[int(cluster_id)] = evidence
            self.cluster_members_[int(cluster_id)] = members["address"].tolist()

        top_features = self._top_feature_names(df, numeric_cols)

        probs = []
        breakdown_col = []
        for idx, row in df.iterrows():
            cluster_id = int(row["cluster_id"])
            evidence = self.cluster_evidence_.get(cluster_id, {})
            breakdown_col.append(evidence)

            if cluster_id == -1:
                local = 0.05
                local += 0.18 * float(np.clip(row.get("pct_funds_from_top_source", 0.0), 0.0, 1.0))
                local += 0.08 * float(np.clip(row.get("burst_ratio", 0.0) * 2.0, 0.0, 1.0))
                local += 0.08 * float(np.clip(1.0 - row.get("gas_price_cv", 0.0), 0.0, 1.0))
                probs.append(float(np.clip(local, 0.0, 0.49)))
                continue

            local = 0.0
            local += 0.15 * float(np.clip(row.get("pct_funds_from_top_source", 0.0), 0.0, 1.0))
            local += 0.10 * float(np.clip(row.get("burst_ratio", 0.0) * 2.0, 0.0, 1.0))
            local += 0.08 * float(np.clip(1.0 - row.get("gas_price_cv", 0.0), 0.0, 1.0))
            score = float(evidence.get("coordination_score", 0.0))
            probs.append(float(np.clip(0.72 * score + local, 0.0, 0.99)))

        out = pd.DataFrame(
            {
                "address": df["address"],
                "cluster_id": df["cluster_id"].astype(int),
                "sybil_probability": np.array(probs, dtype=float),
                "top_3_features": top_features,
                "coordination_breakdown": breakdown_col,
            }
        ).sort_values("address").reset_index(drop=True)

        self.results_ = out
        return out

    def explain_cluster(self, cluster_id: int) -> str:
        """Return a human-readable explanation string for a cluster."""

        cid = int(cluster_id)
        if cid not in self.cluster_evidence_:
            return "Cluster {} not found or labeled as noise.".format(cid)

        e = self.cluster_evidence_[cid]
        size = len(self.cluster_members_.get(cid, []))
        return (
            "Cluster {cid} flagged because temporal cosine={t:.2f}, gas similarity={g:.2f}, "
            "common funding={f:.2f}, sequential activity={s:.2f}, coordination_score={c:.2f}, "
            "size={size}."
        ).format(
            cid=cid,
            t=e.get("temporal_correlation", 0.0),
            g=e.get("gas_similarity", 0.0),
            f=e.get("common_funding", 0.0),
            s=e.get("sequential_activity", 0.0),
            c=e.get("coordination_score", 0.0),
            size=size,
        )

    def _refine_labels(self, df: pd.DataFrame, base_labels: np.ndarray) -> np.ndarray:
        """Use strict quality gates to keep precision high on behavioral clusters."""

        labels = np.full(len(df), -1, dtype=int)
        next_cluster = 0

        # 1) High-confidence groups by shared common funder with coordination gates.
        grouped = []
        for funder, group in df.groupby("common_funder_address"):
            if funder in {"", "none", "nan"}:
                continue
            if len(group) < max(8, self.min_cluster_size):
                continue
            if float(group["burst_ratio"].median()) < 0.07:
                continue
            if float(group["day_of_week_entropy"].median()) < 1.6:
                continue
            if float(group["hour_of_day_entropy"].median()) < 1.2:
                continue
            if float(group["funding_source_count"].median()) < 2.5:
                continue
            grouped.append(group.index.to_numpy())

        grouped = sorted(grouped, key=lambda idx: len(idx), reverse=True)
        for idx in grouped:
            if np.any(labels[idx] != -1):
                continue
            labels[idx] = next_cluster
            next_cluster += 1

        # 2) Add high-quality HDBSCAN clusters that do not overlap funder groups.
        for cluster_id in sorted([int(c) for c in np.unique(base_labels) if int(c) != -1]):
            idx = np.where(base_labels == cluster_id)[0]
            if idx.size < max(5, self.min_cluster_size):
                continue
            if np.any(labels[idx] != -1):
                continue

            group = df.iloc[idx]
            if float(group["burst_ratio"].median()) < 0.12:
                continue
            if float(group["day_of_week_entropy"].median()) < 1.8:
                continue
            if float(group["funding_source_count"].median()) < 2.5:
                continue

            labels[idx] = next_cluster
            next_cluster += 1

        return labels

    def _coordination_breakdown(self, members: pd.DataFrame) -> Dict[str, float]:
        temporal = self._temporal_correlation(members)

        gas_vals = members["gas_price_cv"].values.astype(float) if "gas_price_cv" in members.columns else np.array([0.0])
        gas_variance = float(np.var(gas_vals)) if gas_vals.size else 0.0
        gas_similarity = float(np.clip(np.exp(-12.0 * gas_variance), 0.0, 1.0))

        funders = members["common_funder_address"].fillna("")
        funders = funders.loc[~funders.isin(["", "none", "nan"])]
        if funders.empty:
            common_funding = 0.0
        else:
            common_funding = float(funders.value_counts().iloc[0]) / float(len(members))

        if "first_tx_timestamp" in members.columns:
            first_ts = members["first_tx_timestamp"].values.astype(float)
        else:
            first_ts = np.array([0.0])
        sequential = self._sequential_activity(first_ts)

        coordination = 0.3 * temporal + 0.2 * gas_similarity + 0.3 * common_funding + 0.2 * sequential

        return {
            "temporal_correlation": float(np.clip(temporal, 0.0, 1.0)),
            "gas_similarity": float(np.clip(gas_similarity, 0.0, 1.0)),
            "common_funding": float(np.clip(common_funding, 0.0, 1.0)),
            "sequential_activity": float(np.clip(sequential, 0.0, 1.0)),
            "coordination_score": float(np.clip(coordination, 0.0, 1.0)),
        }

    def _temporal_correlation(self, members: pd.DataFrame) -> float:
        if len(members) < 2:
            return 0.0

        vectors = []
        hour_cols = [f"hour_hist_{i:02d}" for i in range(24)]
        has_hour_cols = all(c in members.columns for c in hour_cols)

        for _, row in members.iterrows():
            if has_hour_cols:
                vec = np.asarray([float(row[c]) for c in hour_cols], dtype=float)
            else:
                hist = row.get("hour_histogram", [])
                vec = np.asarray(hist if isinstance(hist, list) else [], dtype=float)
                if vec.size != 24:
                    vec = np.zeros(24, dtype=float)

            if np.sum(vec) <= 0:
                vec = np.ones(24, dtype=float) / 24.0
            else:
                vec = vec / np.sum(vec)
            vectors.append(vec)

        mat = np.vstack(vectors)
        sims = cosine_similarity(mat)
        upper = sims[np.triu_indices_from(sims, k=1)]
        if upper.size == 0:
            return 0.0
        return float(np.mean(upper))

    def _sequential_activity(self, first_ts: np.ndarray) -> float:
        if first_ts.size < 2:
            return 0.0

        ranks = pd.Series(first_ts).rank(method="average").to_numpy(dtype=float)
        idx = np.arange(1, first_ts.size + 1, dtype=float)
        if np.std(ranks) < 1e-12:
            return 0.0

        corr = float(np.corrcoef(idx, ranks)[0, 1])
        if np.isnan(corr):
            return 0.0
        return float(np.clip(abs(corr), 0.0, 1.0))

    def _top_feature_names(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[List[str]]:
        med = df[numeric_cols].median(axis=0)
        mad = (df[numeric_cols] - med).abs().median(axis=0).replace(0, 1e-9)

        names = []
        for _, row in df.iterrows():
            z = ((row[numeric_cols] - med).abs() / mad).sort_values(ascending=False)
            names.append(z.index[:3].tolist())
        return names


__all__ = ["SybilDetector"]
