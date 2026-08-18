"""超過逸脱度と連続量アウトカム(ΔxT)の関係（計画書10.10節、xT連続量への切り替え）。

成功/失敗という2値ラベルは情報が粗く、7試合という小サンプルでは統計的
検出力が不足している可能性がある（documents/pseudoreplication_response_report.md）。
自前で学習したxTグリッド（scripts/build_xt_grid.py、既存）を使い、各カウンター
イベントの奪取時点から約4秒後までのxT変化量(ΔxT、連続量)を計算し、超過逸脱度
との関係を2値ラベルの場合と比較する。同じ502件のデータをそのまま使い回せる。

実行: uv run python scripts/excess_deviation_xt_correlation.py [path/to/records.pkl]
（省略時はdata/processed/stage3_ensemble.pkl、λ=0.5・アンサンブル版の主結果）
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
XT_GRID_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "xt_grid.pkl"
DEFAULT_RECORDS = Path(__file__).resolve().parent.parent / "data" / "processed" / "stage3_ensemble.pkl"
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0


def xt_lookup(xt_grid: np.ndarray, nx: int, ny: int, x_m: float, y_m: float) -> float:
    x_norm = np.clip(x_m / PITCH_LENGTH, 0, 0.9999)
    y_norm = np.clip(y_m / PITCH_WIDTH, 0, 0.9999)
    cx = min(int(x_norm * nx), nx - 1)
    cy = min(int(y_norm * ny), ny - 1)
    return float(xt_grid[cx, cy])


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RECORDS
    with open(path, "rb") as f:
        records = pickle.load(f)
    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)
    with open(XT_GRID_PATH, "rb") as f:
        xt_data = pickle.load(f)
    nx, ny, xt_grid = xt_data["nx"], xt_data["ny"], xt_data["xT"]

    ball_pos_by_key = {(t.match_id, t.event_id): t.ball_pos for t in trajs}

    rows = []
    for r in records:
        key = (r["match_id"], r["event_id"])
        ball_pos = ball_pos_by_key[key]
        start_ball, target_ball = ball_pos[0], ball_pos[-1]
        if np.isnan(start_ball).any() or np.isnan(target_ball).any():
            continue
        delta_xt = xt_lookup(xt_grid, nx, ny, *target_ball) - xt_lookup(xt_grid, nx, ny, *start_ball)
        rows.append({**r, "delta_xt": delta_xt})

    print(f"=== {path.name}  (n={len(rows)}, xT欠損除外後) ===\n")

    label = np.array([r["label"] for r in rows])
    excess = np.array([r["excess_deviation"] for r in rows])
    recovery_x = np.array([r["recovery_x"] for r in rows])
    dxt = np.array([r["delta_xt"] for r in rows])
    match_ids = np.array([r["match_id"] for r in rows])

    print(f"delta_xT: mean={dxt.mean():.4f} std={dxt.std():.4f} min={dxt.min():.4f} max={dxt.max():.4f}")
    print(f"delta_xT vs label(2値, 参考): point-biserial r={stats.pointbiserialr(label, dxt)[0]:+.3f} "
          f"p={stats.pointbiserialr(label, dxt)[1]:.4f}\n")

    # 生の相関
    r_raw, p_raw = stats.pearsonr(excess, dxt)
    print(f"[連続量ΔxT] 生の相関(交絡無視): r={r_raw:+.3f}  p={p_raw:.4f}")

    # OLS回帰(奪取位置で制御、通常のSE)
    X = np.column_stack([excess, recovery_x])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = sm.add_constant(X)
    ols_naive = sm.OLS(dxt, X).fit()
    beta, p = ols_naive.params[1], ols_naive.pvalues[1]
    print(f"[連続量ΔxT] OLS回帰(奪取位置で制御, naive SE): beta={beta:+.4f}  p={p:.4f}")

    # クラスタロバストSE(試合単位)：疑似反復を最初から考慮
    _, match_codes = np.unique(match_ids, return_inverse=True)
    ols_cluster = sm.OLS(dxt, X).fit(cov_type="cluster", cov_kwds={"groups": match_codes})
    beta_c, p_c = ols_cluster.params[1], ols_cluster.pvalues[1]
    print(f"[連続量ΔxT] OLS回帰(奪取位置で制御, cluster by match n=7): beta={beta_c:+.4f}  p={p_c:.4f}")

    print("\n--- 比較: 2値ラベルでの同じ検定(ロジスティック回帰) ---")
    ymod = sm.Logit(label, X).fit(disp=0)
    print(f"[2値label] naive SE: beta={ymod.params[1]:+.3f}  p={ymod.pvalues[1]:.4f}")
    ymod_c = sm.Logit(label, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": match_codes})
    print(f"[2値label] cluster by match(n=7): beta={ymod_c.params[1]:+.3f}  p={ymod_c.pvalues[1]:.4f}")


if __name__ == "__main__":
    main()
