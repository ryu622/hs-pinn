"""xT（Expected Threat）グリッドを、手持ちのidsse-data(7試合)から自前で学習する。

計画書7節「既存指標との相関」用。Karun Singhの公開ブログにはヒートマップは
あるが、再利用可能な生の数値グリッドは公開されておらず、socceraction等の
主要実装も「利用者が自分の大規模データで都度学習する」設計だった
（詳細はdocuments/phase0_report.md参照）。7試合という小規模データでは
Karun Singh本来の16x12グリッド（192セル）は疎になりすぎるため、
6x4グリッド（24セル）に粗くして、パス・シュートイベントから直接学習する。

自前データで学習する利点：他リーグ・他大会で較正された値を借用する
ミスマッチがなく、このデータセットに対して自己整合的。欠点：7試合分
（パス6062件、シュート171件）は本来のxTモデルが前提とする数万試合規模
より遥かに少なく、特にゴール確率の推定はシュート数が少ないセルで
不安定になりうる（`GOAL_PRIOR`による正則化で緩和）。

実行: uv run python scripts/build_xt_grid.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from kloppy import sportec
from kloppy.domain import Orientation

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]
NX, NY = 6, 4  # 攻撃方向(x) x 横方向(y)
N_ITER = 10
GOAL_PRIOR_SHOTS = 10.0  # ゴール確率のsmoothing用（全体平均へのベイズ的先入観）
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "xt_grid.pkl"


def cell_of(x: float, y: float) -> tuple[int, int]:
    cx = min(int(x * NX), NX - 1)
    cy = min(int(y * NY), NY - 1)
    return cx, cy


def flip_xy(event) -> tuple[float, float]:
    """STATIC_HOME_AWAY正規化後、awayチームのイベントを反転してhome視点(常に+x攻撃)に揃える。"""
    x, y = event.coordinates.x, event.coordinates.y
    if event.team is not None and event.team.ground.value == "away":
        x, y = 1 - x, 1 - y
    return x, y


def main() -> None:
    action_counts = np.zeros((NX, NY))
    shot_counts = np.zeros((NX, NY))
    goal_counts = np.zeros((NX, NY))
    move_counts = np.zeros((NX, NY, NX, NY))  # [from_x, from_y, to_x, to_y]

    n_passes = n_shots = n_goals = 0

    for match_id in MATCH_IDS:
        events = sportec.load_open_event_data(match_id=match_id)
        events = events.transform(to_orientation=Orientation.STATIC_HOME_AWAY)

        for e in events.records:
            if e.coordinates is None or e.team is None:
                continue

            if str(e.event_type) == "EventType.PASS":
                fx, fy = flip_xy(e)
                fc = cell_of(fx, fy)
                action_counts[fc] += 1
                n_passes += 1
                if str(e.result) == "COMPLETE" and e.receiver_coordinates is not None:
                    rx, ry = e.receiver_coordinates.x, e.receiver_coordinates.y
                    if e.team.ground.value == "away":
                        rx, ry = 1 - rx, 1 - ry
                    tc = cell_of(rx, ry)
                    move_counts[fc[0], fc[1], tc[0], tc[1]] += 1

            elif str(e.event_type) == "EventType.SHOT":
                fx, fy = flip_xy(e)
                fc = cell_of(fx, fy)
                action_counts[fc] += 1
                shot_counts[fc] += 1
                n_shots += 1
                if str(e.result) == "GOAL":
                    goal_counts[fc] += 1
                    n_goals += 1

    print(f"n_passes={n_passes} n_shots={n_shots} n_goals={n_goals}")
    print(f"action count per cell: min={action_counts.min():.0f} max={action_counts.max():.0f}")

    total_actions = action_counts.copy()
    total_actions[total_actions == 0] = np.nan  # ゼロ除算防止（今回は全セルaction>0のはず）

    shot_prob = shot_counts / total_actions
    global_goal_rate = n_goals / n_shots
    # ゴール確率：シュート数が少ないセルは全体平均へ縮小するsmoothing
    goal_prob = (goal_counts + GOAL_PRIOR_SHOTS * global_goal_rate) / (shot_counts + GOAL_PRIOR_SHOTS)

    transition_prob = np.zeros((NX, NY, NX, NY))
    for fx in range(NX):
        for fy in range(NY):
            denom = total_actions[fx, fy]
            if np.isnan(denom) or denom == 0:
                continue
            transition_prob[fx, fy] = move_counts[fx, fy] / denom

    # xTの反復計算
    xT = np.zeros((NX, NY))
    for _ in range(N_ITER):
        move_value = np.einsum("ijkl,kl->ij", transition_prob, xT)
        xT = shot_prob * goal_prob + move_value

    print("\nxT grid (行=攻撃方向, 列=横方向, 行0=自陣寄り, 行5=相手ゴール側):")
    print(np.round(xT, 4))

    with open(OUT_PATH, "wb") as f:
        pickle.dump(
            {
                "nx": NX,
                "ny": NY,
                "xT": xT,
                "shot_prob": shot_prob,
                "goal_prob": goal_prob,
                "n_passes": n_passes,
                "n_shots": n_shots,
                "n_goals": n_goals,
            },
            f,
        )
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
