"""合成データでの健全性チェック（計画書7節、SFM/dd-oadの復元テストに相当）。

実際のカウンター軌道1フレームを土台に、既知の方向でセオリー逸脱を人工的に
作り出し（コンパクトネスを崩す／スペース支配を失わせる）、L_tactic
（判定フェーズで使っている定式化そのもの）が意図通りの方向・単調性で
反応するかを確認する。モデル学習は一切関与しない、指標自体のバグ確認。

- コンパクトネス側：攻撃側の重心を中心に、各選手の位置を係数factorで
  拡大・縮小する（factor<1で密集、factor>1で間延び）。
  正解：factorが大きいほどVar/Std/凸包面積は単調に増加するはず。
- スペース支配側：攻撃側フォーメーション全体を、守備側の重心からの相対位置を
  係数push_factorで拡大・縮小する（push_factor=0で守備重心に全員収束、
  push_factor>1で守備重心から一様に離れる）。
  正解：push_factorが大きいほど支配度は単調に増加するはず。

実行: uv run python scripts/synthetic_validity_check.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from hs_pinn.space_control import (
    build_pitch_grid,
    estimate_velocities,
    isotropic_gaussian_dominance,
    velocity_oriented_dominance,
    voronoi_dominance,
)
from hs_pinn.tactic_metrics import (
    convex_hull_area,
    longitudinal_variance,
    pairwise_distance_variance,
)

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
PITCH_MIN = np.array([0.0, 0.0])
PITCH_MAX = np.array([105.0, 68.0])

COMPACT_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
SPACE_PUSH_FACTORS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

COMPACT_FORMULATIONS = {
    "pairwise_distance_variance": pairwise_distance_variance,
    "longitudinal_variance": longitudinal_variance,
    "convex_hull_area": convex_hull_area,
}


def _valid_rows(pos: np.ndarray) -> np.ndarray:
    return ~np.isnan(pos[:, 0])


def perturb_compactness(attack_pos: np.ndarray, factor: float) -> np.ndarray:
    valid = _valid_rows(attack_pos)
    centroid = attack_pos[valid].mean(axis=0)
    out = attack_pos.copy()
    out[valid] = centroid + factor * (attack_pos[valid] - centroid)
    out[valid] = np.clip(out[valid], PITCH_MIN, PITCH_MAX)
    return out


def perturb_space(attack_pos: np.ndarray, defend_pos: np.ndarray, push_factor: float) -> np.ndarray:
    """攻撃側のフォーメーション（選手間の相対配置）を保ったまま、守備側の
    重心を基準に一様拡大・縮小する。push_factor=0で守備重心に全員収束（最悪ケース）、
    push_factor>1で守備の重心から離れる方向に一様に広がる（改善方向）。

    （最初の実装は「各選手を"自分の"最寄りマーカーから引き離す」という
    選手ごとに向きがバラバラな摂動だったが、これはチーム全体の支配面積という
    集約指標とは方向が食い違いうる摂動だったため、コンパクトネス側と同じ
    「基準点からの一様スケーリング」に統一した。）
    """
    a_valid = _valid_rows(attack_pos)
    d_valid = _valid_rows(defend_pos)
    defend_centroid = defend_pos[d_valid].mean(axis=0)

    out = attack_pos.copy()
    out[a_valid] = defend_centroid + push_factor * (attack_pos[a_valid] - defend_centroid)
    out[a_valid] = np.clip(out[a_valid], PITCH_MIN, PITCH_MAX)
    return out


def check_compactness(trajectories: list, frame_idx: int = -1) -> dict:
    results = {name: [] for name in COMPACT_FORMULATIONS}
    for traj in trajectories:
        base = traj.attack_pos[frame_idx]
        if np.isnan(base[:, 0]).all():
            continue
        for name, fn in COMPACT_FORMULATIONS.items():
            values = [fn(perturb_compactness(base, f)) for f in COMPACT_FACTORS]
            if any(np.isnan(v) for v in values):
                continue
            corr, _ = spearmanr(COMPACT_FACTORS, values)
            results[name].append(corr)
    return results


def check_space(
    trajectories: list,
    frame_idx: int = -1,
    sigma: float = 9.0,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> dict:
    formulations = ["voronoi_dominance", "isotropic_gaussian_dominance", "velocity_oriented_dominance"]
    results = {name: [] for name in formulations}
    grid = build_pitch_grid(pitch_length, pitch_width)

    for traj in trajectories:
        base_attack = traj.attack_pos[frame_idx]
        base_defend = traj.defend_pos[frame_idx]
        if np.isnan(base_attack[:, 0]).all() or np.isnan(base_defend[:, 0]).all():
            continue

        attack_vel = estimate_velocities(traj.attack_pos, traj.frame_rate)[frame_idx]
        defend_vel = estimate_velocities(traj.defend_pos, traj.frame_rate)[frame_idx]

        vals_voronoi, vals_iso, vals_vel = [], [], []
        for pf in SPACE_PUSH_FACTORS:
            perturbed = perturb_space(base_attack, base_defend, pf)
            vals_voronoi.append(voronoi_dominance(grid, perturbed, base_defend))
            vals_iso.append(isotropic_gaussian_dominance(grid, perturbed, base_defend, sigma))
            vals_vel.append(
                velocity_oriented_dominance(grid, perturbed, base_defend, attack_vel, defend_vel)
            )

        for name, vals in zip(formulations, [vals_voronoi, vals_iso, vals_vel]):
            if any(np.isnan(v) for v in vals):
                continue
            corr, _ = spearmanr(SPACE_PUSH_FACTORS, vals)
            results[name].append(corr)

    return results


def summarize(results: dict, label: str) -> None:
    print(f"=== {label} ===")
    for name, corrs in results.items():
        corrs = np.array(corrs)
        n = len(corrs)
        n_strong = (corrs > 0.8).sum()
        print(
            f"  {name:32s} n={n:3d}  mean_corr={corrs.mean():+.3f}  "
            f"frac(corr>0.8)={n_strong/n:.2%}" if n else f"  {name:32s} n=0 (skipped)"
        )
    print()


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        trajectories = pickle.load(f)

    print(f"n_trajectories: {len(trajectories)}\n")

    compact_results = check_compactness(trajectories)
    summarize(compact_results, "コンパクトネス（factor↑ → 密集を崩す → 値は増加するはず）")

    space_results = check_space(trajectories)
    summarize(space_results, "スペース支配（push_factor↑ → マーカーから離れる → 値は増加するはず）")


if __name__ == "__main__":
    main()
