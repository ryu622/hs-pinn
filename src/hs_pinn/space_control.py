"""スペース支配（pitch control）の定式化（判定フェーズ用の簡易版）。

計画書2.3節が参照するFernández & Bornn型pitch controlの正確な再現ではなく、
判定フェーズで「安く確認する」ための簡易プロキシとして3通りの定式化を用意する
（判定③：定式化を変えても傾向が安定するかの確認用）。

1. voronoi_dominance            : 最近傍選手が攻撃側であるグリッド点の割合（速度を考慮しない最も単純な幾何指標）
2. isotropic_gaussian_dominance : 等方ガウス影響関数の合計に基づく支配度（速度なし、半径固定）
3. velocity_oriented_dominance  : 進行方向に伸びる異方ガウス影響関数（Fernández & Bornnに寄せた簡易版）

ハイパーパラメータ（影響半径など）は判定フェーズ用の仮の値であり、本実装フェーズで
チューニングが必要になる可能性がある。
"""

from __future__ import annotations

import numpy as np


def build_pitch_grid(
    pitch_length: float, pitch_width: float, nx: int = 22, ny: int = 15
) -> np.ndarray:
    """ピッチ全面を覆う格子点を返す。shape=(nx*ny, 2)。"""
    xs = np.linspace(0.0, pitch_length, nx)
    ys = np.linspace(0.0, pitch_width, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def estimate_velocities(
    positions: np.ndarray, frame_rate: float, half_window_frames: int = 5
) -> np.ndarray:
    """(T, n_players, 2) の位置系列から中心差分で速度(m/s)を推定する。

    卒論のdt_velocity=0.20s（5フレーム差分）に合わせ、既定でも5フレーム幅を使う。
    """
    T = positions.shape[0]
    velocities = np.full_like(positions, np.nan)
    dt = half_window_frames / frame_rate
    for t in range(T):
        t0 = max(0, t - half_window_frames)
        t1 = min(T - 1, t + half_window_frames)
        if t1 == t0:
            continue
        actual_dt = (t1 - t0) / frame_rate
        velocities[t] = (positions[t1] - positions[t0]) / actual_dt
    return velocities


def _valid_rows(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mask = ~np.isnan(arrays[0][:, 0])
    for arr in arrays[1:]:
        mask &= ~np.isnan(arr[:, 0])
    return tuple(arr[mask] for arr in arrays)


def voronoi_dominance(
    grid: np.ndarray,
    attack_pos_t: np.ndarray,
    defend_pos_t: np.ndarray,
) -> float:
    (attack_valid,) = _valid_rows(attack_pos_t)
    (defend_valid,) = _valid_rows(defend_pos_t)
    if len(attack_valid) == 0 or len(defend_valid) == 0:
        return np.nan

    d_attack = np.linalg.norm(grid[:, None, :] - attack_valid[None, :, :], axis=2).min(axis=1)
    d_defend = np.linalg.norm(grid[:, None, :] - defend_valid[None, :, :], axis=2).min(axis=1)
    return float((d_attack < d_defend).mean())


def _isotropic_influence(grid: np.ndarray, positions: np.ndarray, sigma: float) -> np.ndarray:
    diff = grid[:, None, :] - positions[None, :, :]  # (G, n, 2)
    sq_dist = (diff**2).sum(axis=2)
    return np.exp(-0.5 * sq_dist / sigma**2).sum(axis=1)  # (G,)


def isotropic_gaussian_dominance(
    grid: np.ndarray,
    attack_pos_t: np.ndarray,
    defend_pos_t: np.ndarray,
    sigma: float = 9.0,
) -> float:
    (attack_valid,) = _valid_rows(attack_pos_t)
    (defend_valid,) = _valid_rows(defend_pos_t)
    if len(attack_valid) == 0 or len(defend_valid) == 0:
        return np.nan

    net = _isotropic_influence(grid, attack_valid, sigma) - _isotropic_influence(
        grid, defend_valid, sigma
    )
    control = 1.0 / (1.0 + np.exp(-net))
    return float(control.mean())


def _oriented_influence(
    grid: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    sigma_across: float,
    speed_to_sigma_along: float,
    lookahead_s: float,
) -> np.ndarray:
    total = np.zeros(grid.shape[0])
    for pos, vel in zip(positions, velocities):
        speed = np.linalg.norm(vel)
        center = pos + vel * lookahead_s
        if speed < 1e-6:
            cos_t, sin_t = 1.0, 0.0
        else:
            cos_t, sin_t = vel[0] / speed, vel[1] / speed
        sigma_along = sigma_across + speed_to_sigma_along * speed

        diff = grid - center  # (G, 2)
        x_rot = diff[:, 0] * cos_t + diff[:, 1] * sin_t
        y_rot = -diff[:, 0] * sin_t + diff[:, 1] * cos_t
        exponent = -0.5 * ((x_rot / sigma_along) ** 2 + (y_rot / sigma_across) ** 2)
        total += np.exp(exponent)
    return total


def velocity_oriented_dominance(
    grid: np.ndarray,
    attack_pos_t: np.ndarray,
    defend_pos_t: np.ndarray,
    attack_vel_t: np.ndarray,
    defend_vel_t: np.ndarray,
    sigma_across: float = 7.0,
    speed_to_sigma_along: float = 1.5,
    lookahead_s: float = 0.5,
) -> float:
    attack_p, attack_v = _valid_rows(attack_pos_t, attack_vel_t)
    defend_p, defend_v = _valid_rows(defend_pos_t, defend_vel_t)
    if len(attack_p) == 0 or len(defend_p) == 0:
        return np.nan

    net = _oriented_influence(
        grid, attack_p, attack_v, sigma_across, speed_to_sigma_along, lookahead_s
    ) - _oriented_influence(
        grid, defend_p, defend_v, sigma_across, speed_to_sigma_along, lookahead_s
    )
    control = 1.0 / (1.0 + np.exp(-net))
    return float(control.mean())


SPACE_FORMULATIONS = {
    "voronoi_dominance": voronoi_dominance,
    "isotropic_gaussian_dominance": isotropic_gaussian_dominance,
    "velocity_oriented_dominance": velocity_oriented_dominance,
}
