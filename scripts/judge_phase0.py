"""判定①〜③（コンパクトネス・スペース支配、計6定式化）をキャッシュ済み軌道データに対して実行する。

実行: uv run python scripts/judge_phase0.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import stats

from hs_pinn.tactic_metrics import COMPACTNESS_FORMULATIONS, compactness_scalar
from hs_pinn.space_control import (
    SPACE_FORMULATIONS,
    build_pitch_grid,
    estimate_velocities,
    voronoi_dominance,
    isotropic_gaussian_dominance,
    velocity_oriented_dominance,
)

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FRAME_STRIDE = 10  # 0.4秒おきにサンプリングして計算コストを抑える


def space_dominance_scalar(traj, grid, formulation: str) -> float:
    T = traj.attack_pos.shape[0]
    frame_indices = range(0, T, FRAME_STRIDE)

    if formulation == "velocity_oriented_dominance":
        attack_vel = estimate_velocities(traj.attack_pos, traj.frame_rate)
        defend_vel = estimate_velocities(traj.defend_pos, traj.frame_rate)

    values = []
    for t in frame_indices:
        if formulation == "voronoi_dominance":
            v = voronoi_dominance(grid, traj.attack_pos[t], traj.defend_pos[t])
        elif formulation == "isotropic_gaussian_dominance":
            v = isotropic_gaussian_dominance(grid, traj.attack_pos[t], traj.defend_pos[t])
        elif formulation == "velocity_oriented_dominance":
            v = velocity_oriented_dominance(
                grid,
                traj.attack_pos[t],
                traj.defend_pos[t],
                attack_vel[t],
                defend_vel[t],
            )
        else:
            raise ValueError(formulation)
        values.append(v)
    return float(np.nanmean(values))


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)

    labels = np.array([t.label for t in trajs])
    print(f"n = {len(trajs)}, success rate = {labels.mean():.3f}\n")

    grid = build_pitch_grid(PITCH_LENGTH, PITCH_WIDTH)

    all_scores: dict[str, np.ndarray] = {}

    print("=" * 60)
    print("コンパクトネス")
    print("=" * 60)
    for name in COMPACTNESS_FORMULATIONS:
        vals = np.array([compactness_scalar(t.attack_pos, name) for t in trajs])
        all_scores[f"compact:{name}"] = vals
        valid = ~np.isnan(vals)
        r, p = stats.pointbiserialr(labels[valid], vals[valid])
        print(
            f"{name:32s} mean={np.nanmean(vals):8.2f} std={np.nanstd(vals):7.2f} "
            f"corr(label)={r:+.3f} (p={p:.3f})"
        )

    print()
    print("=" * 60)
    print("スペース支配")
    print("=" * 60)
    for name in SPACE_FORMULATIONS:
        vals = np.array([space_dominance_scalar(t, grid, name) for t in trajs])
        all_scores[f"space:{name}"] = vals
        valid = ~np.isnan(vals)
        r, p = stats.pointbiserialr(labels[valid], vals[valid])
        print(
            f"{name:32s} mean={np.nanmean(vals):8.4f} std={np.nanstd(vals):7.4f} "
            f"corr(label)={r:+.3f} (p={p:.3f})"
        )

    print()
    print("=" * 60)
    print("判定③: 全定式化間の相関行列")
    print("=" * 60)
    names = list(all_scores.keys())
    header = " " * 34 + " ".join(f"{n[:10]:>11s}" for n in names)
    print(header)
    for i, ni in enumerate(names):
        row = [f"{ni:34s}"]
        for nj in names:
            a, b = all_scores[ni], all_scores[nj]
            valid = ~(np.isnan(a) | np.isnan(b))
            r, _ = stats.pearsonr(a[valid], b[valid])
            row.append(f"{r:+11.2f}")
        print(" ".join(row))

    out_path = CACHE_PATH.parent / "phase0_scores.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"labels": labels, "scores": all_scores}, f)
    print(f"\nsaved scores to {out_path}")


if __name__ == "__main__":
    main()
