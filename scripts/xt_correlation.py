"""xT（自前学習、scripts/build_xt_grid.py参照）とL_tacticの相関（計画書7節）。

各カウンター候補シーンについて、奪取時点と5秒後のボール位置のxT差分（ΔxT）を
計算し、コンパクトネス・スペース支配（Voronoi占有率）との相関を確認する。
xG（scripts/xg_correlation.py）はシュートを要求するため対象がn=14まで
絞られたが、ΔxTは全イベントに計算できるためn=502で検定できる。

事前に scripts/build_xt_grid.py を実行して data/processed/xt_grid.pkl を
作成しておく必要がある。

実行: uv run python scripts/xt_correlation.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import stats

from hs_pinn.space_control import build_pitch_grid, voronoi_dominance
from hs_pinn.tactic_metrics import compactness_scalar

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
XT_GRID_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "xt_grid.pkl"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "xt_correlation.pkl"

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
FRAME_STRIDE = 10


def xt_lookup(xt_grid: np.ndarray, nx: int, ny: int, x_m: float, y_m: float) -> float:
    x_norm = np.clip(x_m / PITCH_LENGTH, 0, 0.9999)
    y_norm = np.clip(y_m / PITCH_WIDTH, 0, 0.9999)
    cx = min(int(x_norm * nx), nx - 1)
    cy = min(int(y_norm * ny), ny - 1)
    return float(xt_grid[cx, cy])


def space_dominance_scalar(traj, grid: np.ndarray) -> float:
    T = traj.attack_pos.shape[0]
    values = [
        voronoi_dominance(grid, traj.attack_pos[t], traj.defend_pos[t])
        for t in range(0, T, FRAME_STRIDE)
    ]
    return float(np.nanmean(values))


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        trajectories = pickle.load(f)
    with open(XT_GRID_PATH, "rb") as f:
        xt_data = pickle.load(f)

    nx, ny, xt_grid = xt_data["nx"], xt_data["ny"], xt_data["xT"]
    pitch_grid = build_pitch_grid(PITCH_LENGTH, PITCH_WIDTH)

    records = []
    for traj in trajectories:
        start_ball, target_ball = traj.ball_pos[0], traj.ball_pos[-1]
        if np.isnan(start_ball).any() or np.isnan(target_ball).any():
            continue
        delta_xt = xt_lookup(xt_grid, nx, ny, *target_ball) - xt_lookup(xt_grid, nx, ny, *start_ball)
        compact = compactness_scalar(traj.attack_pos, "longitudinal_variance")
        space = space_dominance_scalar(traj, pitch_grid)
        records.append({"label": traj.label, "delta_xt": delta_xt, "compact": compact, "space": space})

    label = np.array([r["label"] for r in records])
    dxt = np.array([r["delta_xt"] for r in records])
    compact = np.array([r["compact"] for r in records])
    space = np.array([r["space"] for r in records])

    print(f"n = {len(records)}")
    print(f"delta_xT: mean={dxt.mean():.4f} std={dxt.std():.4f} min={dxt.min():.4f} max={dxt.max():.4f}\n")

    r1, p1 = stats.pointbiserialr(label, dxt)
    print(f"delta_xT vs success label: r={r1:+.3f} p={p1:.4f}")

    valid_c = ~np.isnan(compact)
    r2, p2 = stats.pearsonr(compact[valid_c], dxt[valid_c])
    print(f"delta_xT vs L_compact(longitudinal_variance): r={r2:+.3f} p={p2:.4f} (n={valid_c.sum()})")

    valid_s = ~np.isnan(space)
    r3, p3 = stats.pearsonr(space[valid_s], dxt[valid_s])
    print(f"delta_xT vs L_space(voronoi_dominance): r={r3:+.3f} p={p3:.4f} (n={valid_s.sum()})")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(records, f)
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
