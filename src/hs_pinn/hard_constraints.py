"""ハード制約層（計画書3.3節）。

選手の身体が物理的に破れない制約（最大加速度・最大速度・ピッチ境界）を、
損失関数のペナルティではなく forward pass 内の射影・積分層として実装する。
学習の状態に関わらず、この層を通過した時点で違反は原理的に発生しない。

制約は計画書の式の通り、加速度・速度とも「ベクトルの大きさ」ではなく
x, y各成分を独立にtanh射影・clipする（計画書3.3節の式をそのまま実装したもの。
対角方向には a_max*sqrt(2) まで出せてしまう点は既知の単純化）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PitchBounds:
    x_min: float = 0.0
    x_max: float = 105.0
    y_min: float = 0.0
    y_max: float = 68.0


class HardConstraintLayer(nn.Module):
    """デコーダが出力する"生の"加速度系列を、物理的に実現可能な軌道へ変換する。

    forward(raw_accel, init_position, init_velocity) -> (positions, velocities, accelerations)

    raw_accel: (..., T, 2)  デコーダの生出力（未来T frame分の加速度）
    init_position, init_velocity: (..., 2)  観測窓の最終フレームでの位置・速度
    戻り値はいずれも (..., T, 2)。
    """

    def __init__(
        self,
        a_max: float = 6.0,
        v_max: float = 9.0,
        dt: float = 0.04,
        pitch_bounds: PitchBounds | None = None,
    ) -> None:
        super().__init__()
        self.a_max = a_max
        self.v_max = v_max
        self.dt = dt
        self.pitch_bounds = pitch_bounds or PitchBounds()

    def forward(
        self, raw_accel: Tensor, init_position: Tensor, init_velocity: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        T = raw_accel.shape[-2]
        pb = self.pitch_bounds

        positions = []
        velocities = []
        accelerations = []

        pos = init_position
        vel = init_velocity

        for t in range(T):
            # 加速度の上限（tanh射影、成分ごと）
            a_t = self.a_max * torch.tanh(raw_accel[..., t, :] / self.a_max)

            # 速度の上限（Euler積分＋clip、成分ごと）
            vel = torch.clamp(vel + a_t * self.dt, -self.v_max, self.v_max)

            # 位置の更新
            pos = pos + vel * self.dt

            # ピッチ境界（clamp）
            x = torch.clamp(pos[..., 0], pb.x_min, pb.x_max)
            y = torch.clamp(pos[..., 1], pb.y_min, pb.y_max)
            pos = torch.stack([x, y], dim=-1)

            accelerations.append(a_t)
            velocities.append(vel)
            positions.append(pos)

        return (
            torch.stack(positions, dim=-2),
            torch.stack(velocities, dim=-2),
            torch.stack(accelerations, dim=-2),
        )


class UnconstrainedIntegrationLayer(nn.Module):
    """フェーズ3の3パターン比較（計画書6.1節）における「制約なし」ベースライン用。

    `HardConstraintLayer`と同じforwardシグネチャ・同じEuler積分の骨格を持つが、
    tanh射影・速度clip・ピッチ境界clampを一切行わない（生の加速度をそのまま積分するだけ）。
    差し替え可能にすることで、3パターン比較の差分が「積分するかどうか」ではなく
    「制約を課すかどうか」だけになるようにしている。
    """

    def __init__(self, dt: float = 0.04) -> None:
        super().__init__()
        self.dt = dt

    def forward(
        self, raw_accel: Tensor, init_position: Tensor, init_velocity: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        T = raw_accel.shape[-2]

        positions = []
        velocities = []
        accelerations = []

        pos = init_position
        vel = init_velocity

        for t in range(T):
            a_t = raw_accel[..., t, :]
            vel = vel + a_t * self.dt
            pos = pos + vel * self.dt

            accelerations.append(a_t)
            velocities.append(vel)
            positions.append(pos)

        return (
            torch.stack(positions, dim=-2),
            torch.stack(velocities, dim=-2),
            torch.stack(accelerations, dim=-2),
        )
