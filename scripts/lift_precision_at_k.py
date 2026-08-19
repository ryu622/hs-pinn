"""Lift/Precision@K評価：dist_B単体 vs 超過逸脱度のベイクオフ（計画書11.3節・11.9節項目1,2）。

超過逸脱度を精密な統計的推定量として扱う設計から、ランキング/発見ツールとして
使う設計へ転換した（11節）。ランキングに使うスコアとして、
  (a) modelB単体の実観測との距離(dist_B)
  (b) 超過逸脱度(dist_B - dist_A)
のどちらが良いかは理論で決め切らず、両方計算してLift/Precision@Kを比較し、
経験的に良い方を主指標として採用する。

MVP方針（11.6節）に従い、信頼区間・ブートストラップ・Leave-One-Match-Out等の
厳密な頑健性チェックは行わない。代わりに、上位K件が何試合・何チームに
またがっているかという診断を添える。

「試合数が少ないため統計的推論の厳密な保証はしていない。発見的なスクリーニング
ツールとして位置づけている」という限界を明記する。

実行: uv run python scripts/lift_precision_at_k.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-WhitneyのU統計量に基づくAUC（sklearn非依存）。"""
    ranks = rankdata(scores)
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    sum_ranks_pos = ranks[labels == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))

RECORDS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "excess_deviation_aggregated.pkl"
OUT_DIR = Path(__file__).resolve().parent.parent / "documents"
K_FRACTIONS = [0.1, 0.2, 0.3, 0.5]


def precision_lift_at_k(scores: np.ndarray, labels: np.ndarray, k_frac: float) -> tuple[int, float, float]:
    n = len(scores)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-scores)
    top_idx = order[:k]
    precision = labels[top_idx].mean()
    base_rate = labels.mean()
    lift = precision / base_rate if base_rate > 0 else float("nan")
    return k, precision, lift


def diagnose_top_k(records: list[dict], scores: np.ndarray, k_frac: float) -> tuple[int, int]:
    n = len(scores)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-scores)
    top_idx = order[:k]
    top_matches = {records[i]["match_id"] for i in top_idx}
    top_teams = {records[i]["defend_team_id"] for i in top_idx}
    return len(top_matches), len(top_teams)


def cumulative_gains(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cum_positives = np.cumsum(sorted_labels)
    frac_events = np.arange(1, len(labels) + 1) / len(labels)
    frac_positives = cum_positives / labels.sum()
    return frac_events, frac_positives


def main() -> None:
    with open(RECORDS_PATH, "rb") as f:
        records = pickle.load(f)

    labels = np.array([r["label"] for r in records])
    dist_b = np.array([r["dist_B"] for r in records])
    excess = np.array([r["excess_deviation"] for r in records])
    n_matches_total = len({r["match_id"] for r in records})
    n_teams_total = len({r["defend_team_id"] for r in records})

    print(f"n={len(records)}  base_rate(success)={labels.mean():.3f}  "
          f"n_matches={n_matches_total}  n_teams={n_teams_total}\n")
    print(f"AUC(dist_B単体)       = {auc_score(labels, dist_b):.3f}")
    print(f"AUC(超過逸脱度)       = {auc_score(labels, excess):.3f}\n")

    for name, scores in [("dist_B単体", dist_b), ("超過逸脱度", excess)]:
        print("=" * 78)
        print(f"スコア: {name}")
        print("=" * 78)
        print(f"{'K%':>5s} {'n_top':>6s} {'precision@K':>12s} {'lift@K':>8s} {'match数':>8s} {'team数':>7s}")
        for kf in K_FRACTIONS:
            k, prec, lift = precision_lift_at_k(scores, labels, kf)
            n_m, n_t = diagnose_top_k(records, scores, kf)
            print(f"{kf*100:4.0f}% {k:6d} {prec:12.3f} {lift:8.2f} {n_m:8d} {n_t:7d}")
        print()

    # Lift chart（累積ゲイン曲線）
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores, color in [("dist_B alone", dist_b, "tab:orange"), ("excess deviation", excess, "tab:blue")]:
        frac_events, frac_positives = cumulative_gains(scores, labels)
        ax.plot(frac_events, frac_positives, label=name, color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.set_xlabel("fraction of events ranked (descending score)")
    ax.set_ylabel("cumulative fraction of successes covered")
    ax.set_title("Cumulative gains: dist_B alone vs excess deviation")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "lift_precision_at_k_cumulative_gains.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
