"""疑似反復への対応（計画書10.10節 項目12、10.9節）：クラスタロバスト標準誤差での再検定。

502件のイベントは同一チーム・同一試合の観測を繰り返し含んでおり、通常の
（クラスタを無視した）ロジスティック回帰は標準誤差を過小評価し、p値を
実際より小さく見せてしまうリスクがある（疑似反復）。

本来は混合効果モデル（選手・チームをランダム効果に入れる）が望ましいが、
実装コストが高い。今回は代わりに、はるかに軽量な「クラスタロバスト標準誤差」
（試合単位・守備側チーム単位でクラスタリング）で標準的な回帰の頑健性を確認する。
係数の点推定は変えず、標準誤差だけをクラスタ内相関を許容する形に補正する。

実行: uv run python scripts/cluster_robust_check.py [path/to/records.pkl]
（省略時はdata/processed/stage3_ensemble.pkl、すなわちλ=0.5・アンサンブル版の主結果）
"""

from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import statsmodels.api as sm

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
DEFAULT_RECORDS = Path(__file__).resolve().parent.parent / "data" / "processed" / "stage3_ensemble.pkl"


def add_defend_team_id(records: list[dict]) -> None:
    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)
    teams_by_match: dict[str, set[str]] = defaultdict(set)
    attack_team_by_key = {}
    for t in trajs:
        teams_by_match[t.match_id].add(t.team_id)
        attack_team_by_key[(t.match_id, t.event_id)] = t.team_id
    for r in records:
        key = (r["match_id"], r["event_id"])
        attack_team = attack_team_by_key[key]
        other_teams = teams_by_match[r["match_id"]] - {attack_team}
        r["defend_team_id"] = next(iter(other_teams)) if other_teams else None


def fit(y, X, cov_type: str, groups=None):
    X = sm.add_constant(X)
    if cov_type == "naive":
        return sm.Logit(y, X).fit(disp=0)
    return sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RECORDS
    with open(path, "rb") as f:
        records = pickle.load(f)
    add_defend_team_id(records)

    labels = np.array([r["label"] for r in records])
    excess = np.array([r["excess_deviation"] for r in records])
    recovery_x = np.array([r["recovery_x"] for r in records])
    match_ids = np.array([r["match_id"] for r in records])
    team_ids = np.array([r["defend_team_id"] for r in records])

    X = np.column_stack([excess, recovery_x])
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    print(f"=== {path.name}  (n={len(records)}, n_matches={len(set(match_ids))}, n_teams={len(set(team_ids))}) ===\n")

    results = {}
    results["naive (クラスタ無視)"] = fit(labels, X, "naive")
    _, match_codes = np.unique(match_ids, return_inverse=True)
    results["cluster by match (n=7)"] = fit(labels, X, "cluster", groups=match_codes)
    _, team_codes = np.unique(team_ids, return_inverse=True)
    results["cluster by defend_team (n=10)"] = fit(labels, X, "cluster", groups=team_codes)

    print(f"{'method':32s} {'beta(excess_dev)':>18s} {'std err':>10s} {'p':>10s}")
    for name, m in results.items():
        beta, se, p = m.params[1], m.bse[1], m.pvalues[1]
        print(f"{name:32s} {beta:18.3f} {se:10.3f} {p:10.4f}")


if __name__ == "__main__":
    main()
