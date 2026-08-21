"""ポスター「手法」パネル用の概念図（modelA/modelBの構成、計画書11節の設計転換を含む）。

実行: uv run python scripts/poster_method_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "Hiragino Sans"

OUT_PATH = Path(__file__).resolve().parent.parent / "documents" / "poster_method_diagram.png"

BOX_FACE = "#eef2f7"
BOX_EDGE = "#33475b"
A_FACE = "#eaf3ea"
A_EDGE = "#3a7d3a"
B_FACE = "#f7ecec"
B_EDGE = "#a6432e"
RANK_FACE = "#fdf3e3"
RANK_EDGE = "#a8722a"


def box(ax, xy, w, h, text, face=BOX_FACE, edge=BOX_EDGE, fontsize=11, weight="normal"):
    x, y = xy
    fb = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(fb)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, zorder=3, linespacing=1.5)
    return (x, y, w, h)


def arrow(ax, p0, p1, color="#333333", lw=1.6, style="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                         linewidth=lw, color=color, zorder=1)
    ax.add_patch(a)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # 1. 入力
    b_input = box(ax, (0.3, 2.6), 2.1, 1.6,
                  "入力\n観測窓（奪取後1秒）\n攻撃側実軌道\n+ 守備側初期状態",
                  fontsize=10.5)

    # 2. 共有バックボーン
    b_backbone = box(ax, (2.9, 2.6), 2.6, 1.6,
                      "PINN Backbone（共有）\nSelf-Attention + GRU\n+ Hard制約層\n(速度・加速度上限, ピッチ境界)",
                      fontsize=10)

    arrow(ax, right(b_input), left(b_backbone))

    # 3. 分岐: modelA / modelB（損失関数だけが異なる）
    b_lossA = box(ax, (6.1, 4.35), 2.2, 0.95, "学習損失\nL_data のみ\n(λ=0)",
                  face=A_FACE, edge=A_EDGE, fontsize=9.5)
    b_lossB = box(ax, (6.1, 1.2), 2.2, 0.95, "学習損失\nL_data + λ・L_theory\n(λ>0, コンパクトネス)",
                  face=B_FACE, edge=B_EDGE, fontsize=9.5)

    bb_top = top(b_backbone)
    bb_bottom = bottom(b_backbone)
    arrow(ax, (bb_top[0], bb_top[1] - 0.0), (6.1, 4.35 + 0.475), color=A_EDGE)
    arrow(ax, (bb_bottom[0], bb_bottom[1]), (6.1, 1.2 + 0.475), color=B_EDGE)

    b_modelA = box(ax, (8.7, 4.35), 2.0, 0.95, "modelA\nデータ駆動ghost",
                   face=A_FACE, edge=A_EDGE, fontsize=10.5, weight="bold")
    b_modelB = box(ax, (8.7, 1.2), 2.0, 0.95, "modelB\nPINNゴースト",
                   face=B_FACE, edge=B_EDGE, fontsize=10.5, weight="bold")
    arrow(ax, right(b_lossA), left(b_modelA), color=A_EDGE)
    arrow(ax, right(b_lossB), left(b_modelB), color=B_EDGE)

    # 4. 実際の守備軌道と比較
    b_real = box(ax, (11.1, 2.6), 1.6, 1.6, "実際の\n守備軌道\n(観測)", fontsize=9.5)
    arrow(ax, top(b_modelA), (11.9, 4.2), color=A_EDGE)
    arrow(ax, (11.9, 4.2), top(b_real), color=A_EDGE)
    arrow(ax, bottom(b_modelB), (11.9, 2.6), color=B_EDGE)

    ax.text(11.9, 4.55, "dist_A", ha="center", fontsize=9, color=A_EDGE, weight="bold")
    ax.text(11.9, 1.85, "dist_B", ha="center", fontsize=9, color=B_EDGE, weight="bold")

    # 5. 下段: 設計転換の注記とランキング評価
    ax.annotate(
        "", xy=(9.7, 0.55), xytext=(9.7, 1.2),
        arrowprops=dict(arrowstyle="-|>", color=RANK_EDGE, lw=1.6),
    )
    b_rank = box(ax, (7.6, -0.55), 3.6, 1.05,
                 "dist_B でランキング\n→ Lift / Precision@K で評価\n（超過逸脱度の差分より高性能）",
                 face=RANK_FACE, edge=RANK_EDGE, fontsize=10)

    fig.text(0.5, 0.965,
              "設計転換：精密な統計的推定量（modelA/modelB差分）から、発見的ランキングツール（dist_B）へ",
              ha="center", fontsize=12.5, weight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
