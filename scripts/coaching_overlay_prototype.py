"""指導現場向け「実際 vs PINNゴースト」重ね合わせ可視化の試作（選手単位の逸脱強調つき）。

11.9節の先、指導現場での活用案（フラグ付きシーンに選手単位の逸脱を可視化する）の
試作版。3者比較（three_way_comparison.py）と異なり、次の2点を変える。

1. 4パネル横並びではなく、REAL(実線)とmodelB PINNゴースト(破線、最大逸脱選手のみ)を
   1枚に重ねて描く
2. 事例選定を「dist_Bランキング上位」ではなく「実際の守備の縦方向の広がりが、
   全体平均(target_std)から最も乖離している事例」にする——3者比較の議論で、
   dist_Bランキング上位×in-sample学習だと実際とほぼ変わらない軌道になりがちだと
   分かったため、modelBの補正が最も視覚化されやすい条件で試作する
3. 運用上の主指標であるλ=0.5・LOMO-CV(該当試合をホールドアウト)で予測する
   (3者比較のλ=8・全データ学習とは異なり、これは実運用と一致した設定)

EVENTSに複数(match_id, event_id)を並べることで、複数事例をまとめて可視化する。
同じ試合の事例は同じホールドアウト学習を使い回して学習コストを抑える。

実行: uv run python scripts/coaching_overlay_prototype.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mplsoccer import Pitch
from torch.utils.data import DataLoader

from hs_pinn.dataset import (
    INPUT_FRAMES,
    CounterAttackDataset,
    collate_samples,
)
from hs_pinn.hard_constraints import HardConstraintLayer, PitchBounds
from hs_pinn.model import TrajectoryBackbone
from hs_pinn.soft_constraints import compactness_loss, compute_target_compactness
from hs_pinn.tactic_metrics import longitudinal_variance

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
OUT_DIR = Path(__file__).resolve().parent.parent / "documents" / "coaching_overlay_images"
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
PITCH_BOUNDS = PitchBounds(0.0, PITCH_LENGTH, 0.0, PITCH_WIDTH)
LAMBDA_OPERATIONAL = 0.5  # 運用上の主指標と同じ設定(11節)
EPOCHS = 30
BATCH_SIZE = 16

# 実観測の守備コンパクトネスがtarget_stdから乖離している上位事例から、
# 試合の多様性・方向の多様性(間延び/過密のどちらも)を確保して選定。
# J03WN1/18226903501176は既にoverlay_..._v2.pngとして作成済みのためここには含めない。
EVENTS = [
    ("J03WQQ", "18240200001012"),  # dev=7.00, real_std=18.66 (間延び)
    ("J03WOH", "18232100001271"),  # dev=6.96, real_std=4.70  (過密、対照)
    ("J03WR9", "18242100000654"),  # dev=6.56, real_std=18.22 (間延び)
]


def train_model(loader, lam: float, target_std: float, seed: int = 0) -> TrajectoryBackbone:
    torch.manual_seed(seed)
    model = TrajectoryBackbone(
        constraint_layer=HardConstraintLayer(a_max=6.0, v_max=9.0, dt=1 / 25, pitch_bounds=PITCH_BOUNDS),
        predict_side="defend",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        for batch in loader:
            pos, _, _ = model(batch)
            target = batch["target_defend_pos"].permute(0, 2, 1, 3)
            mask = batch["defend_mask"]
            diff = (pos - target).norm(dim=-1)
            loss = diff[mask].mean()
            if lam > 0:
                loss = loss + lam * compactness_loss(pos, mask, target_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def plot_event(match_id: str, event_id: str, model_b, target_std: float, holdout_loader, traj_by_key) -> Path:
    key = (match_id, event_id)
    batch = None
    for b in holdout_loader:
        if b["match_id"][0] == match_id and b["event_id"][0] == event_id:
            batch = b
            break
    if batch is None:
        raise RuntimeError(f"event not found: {key}")

    with torch.no_grad():
        pos_b, _, _ = model_b(batch)

    defend_mask = batch["defend_mask"][0].numpy()
    # target_defend_posはバッチ内(T, n_defend, 2)で格納されているため、
    # モデル出力(n_defend, T, 2)と揃えるためpermuteする(stage3_ensemble.py等と同じ規約)
    real_defend = batch["target_defend_pos"].permute(0, 2, 1, 3)[0].numpy()  # (n_defend, T, 2)
    pred_b = pos_b[0].numpy()  # (n_defend, T, 2)
    attack_real = batch["target_attack_pos"][0].numpy()  # (T, n_attack, 2)
    ball_pos = traj_by_key[key].ball_pos[INPUT_FRAMES:]

    real_std_series = [
        np.sqrt(v) for v in (longitudinal_variance(real_defend[:, t][defend_mask]) for t in range(real_defend.shape[1]))
        if not np.isnan(v)
    ]
    real_std = float(np.mean(real_std_series))
    print(f"[{match_id}/{event_id}] real defend longitudinal std = {real_std:.2f} (target = {target_std:.2f})")

    n_defend = real_defend.shape[0]
    final_dist = np.full(n_defend, np.nan)
    mean_dist = np.full(n_defend, np.nan)
    for i in range(n_defend):
        if not defend_mask[i]:
            continue
        final_dist[i] = np.linalg.norm(real_defend[i, -1] - pred_b[i, -1])
        mean_dist[i] = np.linalg.norm(real_defend[i] - pred_b[i], axis=-1).mean()
    worst_player = int(np.nanargmax(mean_dist))
    print("  per-player mean distance (real vs modelB ghost):")
    for i in range(n_defend):
        if defend_mask[i]:
            marker = "  <-- largest deviation" if i == worst_player else ""
            print(f"    player {i}: mean_dist={mean_dist[i]:.2f}m final_dist={final_dist[i]:.2f}m{marker}")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, line_color="black")
    pitch.draw(ax=ax)

    for i in range(attack_real.shape[1]):
        xs, ys = attack_real[:, i, 0], attack_real[:, i, 1]
        if (xs == 0).all() and (ys == 0).all():
            continue
        ax.plot(xs, ys, color="crimson", alpha=0.75, linewidth=2.0, zorder=1,
                 label="attack (real, context)" if i == 0 else None)
        ax.scatter(xs[-1], ys[-1], color="crimson", s=60, zorder=2, marker="^", alpha=0.9)

    first_real_label_done = False
    for i in range(n_defend):
        if not defend_mask[i]:
            continue
        is_worst = i == worst_player
        real_lw = 2.6 if is_worst else 1.6
        xs_r, ys_r = real_defend[i, :, 0], real_defend[i, :, 1]
        ax.plot(xs_r, ys_r, color="royalblue", linewidth=real_lw, linestyle="-", zorder=4,
                 label="REAL defense" if not first_real_label_done else None)
        first_real_label_done = True
        ax.scatter(xs_r[0], ys_r[0], color="royalblue", s=55, zorder=5, marker="o")
        ax.scatter(xs_r[-1], ys_r[-1], color="royalblue", s=90, zorder=5, marker="^")

        if is_worst:
            xs_b, ys_b = pred_b[i, :, 0], pred_b[i, :, 1]
            ax.plot(xs_b, ys_b, color="darkorange", linewidth=2.4, linestyle="--", zorder=4,
                     label="PINN ghost (theory-compliant)")
            ax.scatter(xs_b[0], ys_b[0], color="darkorange", s=55, zorder=5, marker="o", facecolors="none")
            ax.scatter(xs_b[-1], ys_b[-1], color="darkorange", s=90, zorder=5, marker="^", facecolors="none")
            mid = len(xs_r) // 2
            ax.annotate(
                f"largest deviation\n{mean_dist[i]:.1f}m",
                xy=(xs_r[mid], ys_r[mid]), xytext=(xs_r[mid] + 6, ys_r[mid] + 8),
                fontsize=10, color="crimson", weight="bold",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
            )

    ax.plot(ball_pos[:, 0], ball_pos[:, 1], color="black", linewidth=1.8, linestyle=":", zorder=3, alpha=0.9,
             label="ball")
    ax.scatter(ball_pos[0, 0], ball_pos[0, 1], color="black", s=45, zorder=6, marker="o", facecolors="none")
    ax.scatter(ball_pos[-1, 0], ball_pos[-1, 1], color="black", s=45, zorder=6, marker="^")

    ax.set_title(
        f"{match_id} / {event_id}\n"
        f"REAL defense vs PINN ghost (lambda={LAMBDA_OPERATIONAL}, LOMO-CV)  "
        f"defend longitudinal spread: real={real_std:.1f}m / target={target_std:.1f}m",
        fontsize=11,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9, frameon=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"overlay_{match_id}_{event_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")
    return out_path


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        all_trajs = pickle.load(f)

    matches_needed = sorted({m for m, _ in EVENTS})
    for holdout_match in matches_needed:
        train_trajs = [t for t in all_trajs if t.match_id != holdout_match]
        holdout_trajs = [t for t in all_trajs if t.match_id == holdout_match]

        train_ds = CounterAttackDataset(train_trajs)
        holdout_ds = CounterAttackDataset(holdout_trajs)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
        holdout_loader = DataLoader(holdout_ds, batch_size=1, shuffle=False, collate_fn=collate_samples)

        target_std = compute_target_compactness(train_trajs, INPUT_FRAMES, side="defend")
        print(f"training modelB (lambda={LAMBDA_OPERATIONAL}, holdout match={holdout_match}, target_std={target_std:.2f})...")
        model_b = train_model(train_loader, lam=LAMBDA_OPERATIONAL, target_std=target_std)

        traj_by_key = {(t.match_id, t.event_id): t for t in holdout_trajs}
        for match_id, event_id in EVENTS:
            if match_id != holdout_match:
                continue
            plot_event(match_id, event_id, model_b, target_std, holdout_loader, traj_by_key)


if __name__ == "__main__":
    main()
