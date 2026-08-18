"""ケーススタディ用の代表シーン選定・可視化（計画書6.1節・7節）。

L_compact（縦方向の分散、高いほど間延び）とL_space（Voronoi占有率、
低いほど支配できていない）を組み合わせた「セオリーからの逸脱度」指標で、
成功/失敗 × 逸脱度high/lowの2x2で代表シーンを選び、選手の軌道をピッチ図に
プロットする。恣意的な選定に見えないよう、選定基準はここに明記する。

選定基準：
- 逸脱度スコア = zscore(L_compact) - zscore(L_space)
  （間延びしている「かつ/または」スペースを支配できていないほど高スコア）
- 成功イベント・失敗イベントそれぞれの中で、逸脱度スコアが最大/最小の1件を選ぶ
  （calibrationなし、単純に分布の端を取るルールベースの選定）

実行: uv run python scripts/plot_case_studies.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch

from hs_pinn.space_control import build_pitch_grid, voronoi_dominance
from hs_pinn.tactic_metrics import compactness_scalar

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "case_studies"
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
FRAME_STRIDE = 10


def space_dominance_scalar(traj, grid: np.ndarray) -> float:
    T = traj.attack_pos.shape[0]
    values = [
        voronoi_dominance(grid, traj.attack_pos[t], traj.defend_pos[t])
        for t in range(0, T, FRAME_STRIDE)
    ]
    return float(np.nanmean(values))


def plot_event(traj, deviation_score: float, compact: float, space: float, tag: str, tag_en: str) -> Path:
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, line_color="black")
    fig, ax = pitch.draw(figsize=(10, 6.8))

    T = traj.attack_pos.shape[0]

    for i in range(traj.attack_pos.shape[1]):
        xs, ys = traj.attack_pos[:, i, 0], traj.attack_pos[:, i, 1]
        if np.isnan(xs).all():
            continue
        ax.plot(xs, ys, color="crimson", alpha=0.6, linewidth=1.5, zorder=2)
        ax.scatter(xs[0], ys[0], color="crimson", s=60, zorder=3, marker="o")
        ax.scatter(xs[-1], ys[-1], color="crimson", s=90, zorder=3, marker="^")

    for i in range(traj.defend_pos.shape[1]):
        xs, ys = traj.defend_pos[:, i, 0], traj.defend_pos[:, i, 1]
        if np.isnan(xs).all():
            continue
        ax.plot(xs, ys, color="royalblue", alpha=0.4, linewidth=1.2, zorder=1)
        ax.scatter(xs[0], ys[0], color="royalblue", s=40, zorder=2, marker="o")
        ax.scatter(xs[-1], ys[-1], color="royalblue", s=60, zorder=2, marker="^")

    bx, by = traj.ball_pos[:, 0], traj.ball_pos[:, 1]
    ax.plot(bx, by, color="black", linewidth=1.0, linestyle="--", zorder=4)
    ax.scatter(bx[0], by[0], color="gold", edgecolor="black", s=100, zorder=5, marker="*", label="Recovery (t=0)")
    ax.scatter(bx[-1], by[-1], color="black", s=60, zorder=5, marker="X", label="+5s")

    label_str = "SUCCESS" if traj.label == 1 else "FAILURE"
    title = (
        f"{tag_en}\n{traj.match_id} / {traj.event_id}  result={label_str}\n"
        f"L_compact={compact:.1f}  L_space(Voronoi)={space:.3f}  deviation_score={deviation_score:+.2f}"
    )
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{tag}_{traj.match_id}_{traj.event_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)

    grid = build_pitch_grid(PITCH_LENGTH, PITCH_WIDTH)

    compacts, spaces = [], []
    for t in trajs:
        compacts.append(compactness_scalar(t.attack_pos, "longitudinal_variance"))
        spaces.append(space_dominance_scalar(t, grid))
    compacts, spaces = np.array(compacts), np.array(spaces)

    z_compact = (compacts - np.nanmean(compacts)) / np.nanstd(compacts)
    z_space = (spaces - np.nanmean(spaces)) / np.nanstd(spaces)
    deviation = z_compact - z_space  # 高いほど「間延びしていて、かつ/またはスペース支配もできていない」

    labels = np.array([t.label for t in trajs])

    selections = {
        "成功×高逸脱(個人技での打開候補)": ("success_high_deviation (individual skill?)", np.where(labels == 1, deviation, -np.inf).argmax()),
        "成功×低逸脱(セオリー通り候補)": ("success_low_deviation (textbook?)", np.where(labels == 1, deviation, np.inf).argmin()),
        "失敗×高逸脱(組織崩壊候補)": ("failure_high_deviation (breakdown?)", np.where(labels == 0, deviation, -np.inf).argmax()),
        "失敗×低逸脱(セオリー通りだが失敗)": ("failure_low_deviation (textbook but failed)", np.where(labels == 0, deviation, np.inf).argmin()),
    }

    print(f"{'区分':40s} {'match/event':30s} {'compact':>8s} {'space':>7s} {'偏差':>6s}")
    for tag, (tag_en, idx) in selections.items():
        t = trajs[idx]
        path = plot_event(t, deviation[idx], compacts[idx], spaces[idx], tag, tag_en)
        print(f"{tag:40s} {t.match_id}/{t.event_id:15s} {compacts[idx]:8.1f} {spaces[idx]:7.3f} {deviation[idx]:+6.2f}")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
