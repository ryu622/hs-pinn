"""L_tactic（コンパクトネス・スペース支配）の定式化。

判定フェーズ（判定①〜③）向けに、モデル学習なしで観測軌道へ直接計算できる
関数群を提供する。判定③（定式化を変えても傾向が安定するか）に備え、
コンパクトネスは3通りの定式化を用意する。

スペース支配（pitch control）は別途実装予定（未実装）。

【改訂履歴1】初期実装では`centroid_distance_variance`（全方位の重心距離分散）を
採用していたが、phase0_report.md 6節の追加検証で「全方位」より
「攻撃方向（縦）のみ」の分散の方が成功率との相関が明確に強い
（r=0.075→0.185）ことが判明したため、`longitudinal_variance`に差し替えた。

【改訂履歴2】判定フェーズ（判定②③、`judge_phase0.py`等）では長らく
`compactness_scalar`の生の値（分散・面積そのもの）を「逸脱度」として
成功率との相関に使っていたが、計画書2.3節のL_compactの定義は
$|\text{Var}-\text{目標値}|$という目標値からの乖離であり、生の値とは
異なる量だった。`compute_compactness_target` / `compactness_deviation`を
追加し、観測データ全体から算出した目標値との乖離を「逸脱度」として
使うよう修正した（詳細はphase0_report.md参照）。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


def _valid_positions(positions_t: np.ndarray) -> np.ndarray:
    """(n_players, 2) からNaN行を除いたものを返す。"""
    mask = ~np.isnan(positions_t[:, 0])
    return positions_t[mask]


def pairwise_distance_variance(positions_t: np.ndarray) -> float:
    """定式化①：選手間距離（全ペア）の分散。計画書2.3節の定義そのもの。"""
    valid = _valid_positions(positions_t)
    if len(valid) < 2:
        return np.nan
    return float(pdist(valid).var())


def centroid_distance_variance(positions_t: np.ndarray) -> float:
    """（旧定式化②、現在は未使用）チーム重心からの距離の分散（stretch index系）。
    全方位を等価に扱うため、縦の間延びと横への広がりを区別できず、
    成功率との相関が弱かった（r=-0.040）。`longitudinal_variance`に差し替え済み。
    """
    valid = _valid_positions(positions_t)
    if len(valid) < 2:
        return np.nan
    centroid = valid.mean(axis=0)
    dists = np.linalg.norm(valid - centroid, axis=1)
    return float(dists.var())


def longitudinal_variance(positions_t: np.ndarray) -> float:
    """定式化②：攻撃方向（x軸、常に+xが攻撃方向になるよう正規化済み）のみの
    分散。「間延び」は主に縦方向（前後）を指すという解釈に基づく操作化。
    横方向の広がり（幅を使うこと）はコンパクトネス違反とみなさない。
    """
    valid = _valid_positions(positions_t)
    if len(valid) < 2:
        return np.nan
    return float(valid[:, 0].var())


def convex_hull_area(positions_t: np.ndarray) -> float:
    """定式化③：チームの占有面積（凸包の面積、m^2）。距離の分散とは異なる
    幾何学的な観点（広がり方ではなく占有領域そのもの）でコンパクトネスを捉える。
    """
    valid = _valid_positions(positions_t)
    if len(valid) < 3:
        return np.nan
    return float(ConvexHull(valid).volume)  # 2Dでは`volume`が面積を表す


COMPACTNESS_FORMULATIONS = {
    "pairwise_distance_variance": pairwise_distance_variance,
    "longitudinal_variance": longitudinal_variance,
    "convex_hull_area": convex_hull_area,
}


def compactness_timeseries(attack_pos: np.ndarray, formulation: str) -> np.ndarray:
    """attack_pos: (T, n_players, 2) -> (T,) の時系列。"""
    fn = COMPACTNESS_FORMULATIONS[formulation]
    T = attack_pos.shape[0]
    return np.array([fn(attack_pos[t]) for t in range(T)])


def compactness_scalar(attack_pos: np.ndarray, formulation: str) -> float:
    """時間窓全体を平均した、イベント1件あたりのコンパクトネス値（生の値）。"""
    series = compactness_timeseries(attack_pos, formulation)
    return float(np.nanmean(series))


def compute_compactness_target(trajectories: list, formulation: str) -> float:
    """判定フェーズ用の目標値：観測データ全体（全イベント）における
    `compactness_scalar`（生の値）の平均。計画書2.3節の$|\\text{Var}-\\text{目標値}|$の
    「目標値」に相当する（「チームが典型的にはこの程度の広がりを持つ」という
    観測データからの校正値）。

    ※ `soft_constraints.py`の`compute_target_compactness`とは別物。あちらは
    学習損失用にtrain split・予測対象区間（target window）・標準偏差(std)で
    校正しており、ここでは判定フェーズ用に全データ・全窓・生の値（分散等）で
    校正している。目的が異なるため意図的に別関数にしている。
    """
    values = [compactness_scalar(t.attack_pos, formulation) for t in trajectories]
    return float(np.nanmean(values))


def compactness_deviation(attack_pos: np.ndarray, formulation: str, target: float) -> float:
    """L_compact = |生の値 - 目標値|（計画書2.3節の定義通り）。"""
    raw = compactness_scalar(attack_pos, formulation)
    if np.isnan(raw):
        return np.nan
    return abs(raw - target)
