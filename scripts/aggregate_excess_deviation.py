"""超過逸脱度の集計（計画書10.7節）：試合・守備側チーム別。

λ=0.5・5シードアンサンブル版（data/processed/stage3_ensemble.pkl）を対象に、
守備側チーム・試合単位で集計する。守備側チームIDは`CounterTrajectory.team_id`
（攻撃側）には直接入っていないため、各試合に登場する2チームのうち
攻撃側でない方として導出する。

実行: uv run python scripts/aggregate_excess_deviation.py
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
ENSEMBLE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "stage3_ensemble.pkl"


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)
    with open(ENSEMBLE_PATH, "rb") as f:
        records = pickle.load(f)

    # 試合ごとの登場チーム（攻撃側として）を集める→守備側チームを導出
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

    # === 試合単位 ===
    print("=" * 70)
    print("試合単位の集計")
    print("=" * 70)
    by_match: dict[str, list] = defaultdict(list)
    for r in records:
        by_match[r["match_id"]].append(r)

    print(f"{'match_id':10s} {'n':>4s} {'success_rate':>13s} {'excess_dev_mean':>16s} {'excess_dev_std':>15s}")
    for m, recs in sorted(by_match.items()):
        labels = np.array([r["label"] for r in recs])
        excess = np.array([r["excess_deviation"] for r in recs])
        print(f"{m:10s} {len(recs):4d} {labels.mean():13.3f} {excess.mean():16.3f} {excess.std():15.3f}")

    # === 守備側チーム単位 ===
    print()
    print("=" * 70)
    print("守備側チーム単位の集計（n<30のチームは参考値、疑似反復に留意）")
    print("=" * 70)
    by_team: dict[str, list] = defaultdict(list)
    for r in records:
        if r["defend_team_id"] is not None:
            by_team[r["defend_team_id"]].append(r)

    print(f"{'defend_team_id':16s} {'n':>4s} {'opp_success_rate':>17s} {'excess_dev_mean':>16s} {'excess_dev_std':>15s}")
    team_rows = []
    for team, recs in by_team.items():
        labels = np.array([r["label"] for r in recs])
        excess = np.array([r["excess_deviation"] for r in recs])
        team_rows.append((team, len(recs), labels.mean(), excess.mean(), excess.std()))
    team_rows.sort(key=lambda x: -x[3])
    for team, n, succ_rate, exc_mean, exc_std in team_rows:
        print(f"{team:16s} {n:4d} {succ_rate:17.3f} {exc_mean:16.3f} {exc_std:15.3f}")

    # チーム平均excess_deviationと、そのチームが守っている時の相手成功率の関係（n=10チーム、参考程度）
    from scipy import stats

    means = np.array([row[3] for row in team_rows])
    succ_rates = np.array([row[2] for row in team_rows])
    if len(means) >= 3:
        r, p = stats.pearsonr(means, succ_rates)
        print(f"\n(参考) チーム平均超過逸脱度 vs 被カウンター成功率: r={r:+.3f} p={p:.4f} (n={len(means)}チーム)")

    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "excess_deviation_aggregated.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
