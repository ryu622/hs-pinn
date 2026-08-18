"""ソフト制約（計画書2.3節・3.4節）：予測軌道に対する L_compact, L_space の微分可能な実装。

判定フェーズ（`tactic_metrics.py`, `space_control.py`）はnumpyベースで実データに
直接計算するためのものだったが、学習時はモデルの予測軌道（勾配が流れる
`torch.Tensor`）に対して計算する必要があるため、ここでtorchで再実装する。

【設計判断】
- L_compact: 判定フェーズ（phase0_report.md 6節）で「全方位」より「攻撃方向(縦)限定」の
  分散の方が成功率との相関が明確に強いと分かったため、その定式化（`longitudinal_variance`）
  を踏襲する。目標値は学習データの実軌道の統計から計算する（`compute_target_compactness`）。
  ただし損失としては分散（m^2、値が数十〜のオーダー）ではなく**標準偏差（m）**の差で計算する。
  L_data（ADE、m単位で1〜3程度）と単位・スケールを揃えないと、同じλでも
  L_compactだけが支配的になり、精度を無視して分散だけ合わせにいく学習になってしまうため
  （最初の100エポック学習で実際に発生した問題。詳細はdocuments/phase0_report.md等の実験ログ参照）。
- L_space: 判定フェーズの3定式化のうち`voronoi_dominance`は最近傍判定（argmin/しきい値）を
  含み微分不可能なため学習損失には使えない。等方ガウス版（`isotropic_gaussian_dominance`）を
  採用する（速度配向版は速度0付近で回転角が特異点になるため、まずは等方版から）。
  ただしピッチ全面（22×15グリッド）で単純平均すると、大半の点は選手から遠く両チームの
  影響が共にほぼゼロ（sigmoid(0)=0.5固定）になり、選手配置を変えても平均値がほとんど動かない
  という問題が実際の学習で発生した。対策として、いずれかの選手から`near_radius`以内の
  グリッド点だけを対象に平均を取る（`_relevant_grid_mask`）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from hs_pinn.tactic_metrics import longitudinal_variance
from hs_pinn.trajectories import CounterTrajectory


def compactness_loss(
    attack_pos: Tensor,
    attack_mask: Tensor,
    target_std: float,
    frame_stride: int = 5,
    eps: float = 1e-6,
) -> Tensor:
    """L_compact = |Std(攻撃方向の座標) - 目標値|（計画書2.3節の式の分散を標準偏差に変更）。

    分散ではなく標準偏差（m単位）を使うのは、L_data（ADE、m単位）とスケールを揃え、
    同じλでの重み付けが意味を持つようにするため（本モジュールdocstring参照）。

    attack_pos: (B, n_attack, T, 2)。HardConstraintLayerの出力と同じ並び。
    attack_mask: (B, n_attack) bool。
    """
    pos = attack_pos[:, :, ::frame_stride, :]  # (B, n_attack, T', 2)
    mask = attack_mask.float().unsqueeze(-1)  # (B, n_attack, 1)
    n_valid = mask.sum(dim=1)  # (B, 1)

    x = pos[..., 0]  # (B, n_attack, T')
    mean_x = (x * mask).sum(dim=1, keepdim=True) / n_valid.unsqueeze(1)  # (B, 1, T')
    var_x = (((x - mean_x) ** 2) * mask).sum(dim=1) / n_valid  # (B, T')
    std_x = torch.sqrt(var_x + eps)

    return (std_x - target_std).abs().mean()


def _pairwise_sq_dist(entities: Tensor, grid: Tensor) -> Tensor:
    """entities: (B, n, T', 2), grid: (G, 2) -> (B, T', n, G) 各選手-格子点間の距離^2"""
    e = entities.permute(0, 2, 1, 3).unsqueeze(3)  # (B, T', n, 1, 2)
    g = grid.view(1, 1, 1, -1, 2)  # (1, 1, 1, G, 2)
    return ((e - g) ** 2).sum(dim=-1)  # (B, T', n, G)


def _gaussian_influence_from_sq_dist(sq_dist: Tensor, mask: Tensor, sigma: float) -> Tensor:
    """sq_dist: (B, T', n, G), mask: (B, n) -> (B, T', G)"""
    influence = torch.exp(-0.5 * sq_dist / sigma**2)
    influence = influence * mask.view(mask.shape[0], 1, mask.shape[1], 1).float()
    return influence.sum(dim=2)  # (B, T', G)


def _relevant_grid_mask(sq_dist_a: Tensor, sq_dist_d: Tensor, attack_mask: Tensor, defend_mask: Tensor, near_radius: float) -> Tensor:
    """いずれかの選手（攻撃・守備どちらでも）から near_radius 以内にあるグリッド点をTrueとする。
    sq_dist_*: (B, T', n, G)。無効選手（mask=False）は距離を+infにして最小距離計算から除外する。
    戻り値: (B, T', G) bool
    """
    inf = torch.finfo(sq_dist_a.dtype).max
    invalid_a = (~attack_mask.bool()).view(attack_mask.shape[0], 1, attack_mask.shape[1], 1)
    invalid_d = (~defend_mask.bool()).view(defend_mask.shape[0], 1, defend_mask.shape[1], 1)
    min_sq_dist_a = sq_dist_a.masked_fill(invalid_a, inf).min(dim=2).values  # (B, T', G)
    min_sq_dist_d = sq_dist_d.masked_fill(invalid_d, inf).min(dim=2).values
    min_sq_dist = torch.minimum(min_sq_dist_a, min_sq_dist_d)
    return min_sq_dist <= near_radius**2


def space_control_loss(
    attack_pos: Tensor,
    defend_pos: Tensor,
    attack_mask: Tensor,
    defend_mask: Tensor,
    grid: Tensor,
    sigma: float = 9.0,
    frame_stride: int = 10,
    near_radius: float = 15.0,
    pitch_area: float = 105.0 * 68.0,
    eps: float = 1e-6,
) -> Tensor:
    """L_space = -mean(sqrt(選手近傍でのPitchControl_attackの支配面積))。

    計画書2.3節の定義（$-\\sum_{(x,y)}$PitchControl、格子点上のリーマン和）を踏襲しつつ、
    2点補正している。
    (1) ピッチ全面を対象にすると、選手から離れた大半の格子点で両チームの影響がほぼゼロになり
        sigmoid(0)=0.5に張り付くため、選手配置を変えても値がほとんど動かない
        （本モジュールdocstring参照）。→ いずれかの選手からnear_radius以内の格子点だけを対象にする。
    (2) 対象点の"平均"を取ると、複数選手が近接してガウス影響を重ね合わせる（＝密集する）方が
        1点あたりの支配値が高くなり、有利になってしまう（計画書が意図する「広く支配する」の逆）。
        → 平均ではなくcell_area加重の"面積和"（m^2、リーマン和としての本来の$\\sum$に近い）を使い、
        広い範囲を（値は低くても）支配する方が合計値が大きくなるようにする。
        その上でL_data（ADE, m単位）とスケールを揃えるため平方根を取り、線形の長さ(m)に換算する
        （L_compactをVarからStdに直した時と同じ理由、本モジュールdocstring参照）。

    attack_pos: (B, n_attack, T, 2), defend_pos: (B, n_defend, T, 2)（いずれもHardConstraintLayer出力の並び）
    grid: (G, 2) ピッチ全面の格子点（`space_control.build_pitch_grid`をtorch化したもの）
    """
    a = attack_pos[:, :, ::frame_stride, :]
    d = defend_pos[:, :, ::frame_stride, :]

    sq_dist_a = _pairwise_sq_dist(a, grid)  # (B, T', n_attack, G)
    sq_dist_d = _pairwise_sq_dist(d, grid)  # (B, T', n_defend, G)

    total_a = _gaussian_influence_from_sq_dist(sq_dist_a, attack_mask, sigma)  # (B, T', G)
    total_d = _gaussian_influence_from_sq_dist(sq_dist_d, defend_mask, sigma)  # (B, T', G)

    control = torch.sigmoid(total_a - total_d)  # (B, T', G)
    relevant = _relevant_grid_mask(sq_dist_a, sq_dist_d, attack_mask, defend_mask, near_radius)  # (B, T', G)

    cell_area = pitch_area / grid.shape[0]
    area_per_bt = (control * relevant.float()).sum(dim=2) * cell_area  # (B, T'), m^2
    linear_extent = torch.sqrt(area_per_bt + eps)  # (B, T'), m

    space_dominance = linear_extent.mean()
    return -space_dominance


def compute_target_compactness(
    trajectories: list[CounterTrajectory], input_frames: int, side: str = "attack"
) -> float:
    """学習データ（実軌道）の予測対象区間（input_frames以降）における
    縦方向の標準偏差（`longitudinal_variance`の平方根）の平均値を、
    L_compactの目標値として計算する（`compactness_loss`と単位を揃えるため）。

    `side`："attack"（既定）または"defend"。10節（守備側予測への転換）対応。
    """
    if side not in ("attack", "defend"):
        raise ValueError(side)
    values = []
    for traj in trajectories:
        pos = traj.attack_pos if side == "attack" else traj.defend_pos
        target_window = pos[input_frames:]
        for t in range(target_window.shape[0]):
            v = longitudinal_variance(target_window[t])
            if not np.isnan(v):
                values.append(np.sqrt(v))
    return float(np.mean(values))
