"""modelB単体のサニティチェック（計画書10.8節）。

超過逸脱度の計算に使う前に、modelB（理論駆動ghost）の出力が本当に
「セオリーに忠実な、もっともらしい守備」になっているかを目視確認する。
数値的な退化解チェック（段階2）は選手間距離の崩壊は検出できるが、
「不自然だが崩壊はしていない」形状までは検出できないため、別途必要。

実データ全件でmodelA（λ=0）・modelB（λ=0.5、複数シード確認で最も頑健だった
設定。documents/stage3_multiseed_report.md参照。当初はλ=8で実施していたが、
その結果は単一シードの当たりだったと判明したためλ=0.5に変更）を学習し、
いくつかの代表イベントについて「実際の守備」「modelA予測」「modelB予測」を
並べてピッチ図にプロットする。

実行: uv run python scripts/sanity_check_modelB.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
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

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "modelB_sanity_check_lam0.5"
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
PITCH_BOUNDS = PitchBounds(0.0, PITCH_LENGTH, 0.0, PITCH_WIDTH)
EPOCHS = 30
BATCH_SIZE = 16


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


def plot_comparison(event_idx: int, real_defend, pred_a, pred_b, attack_real, defend_mask, ball_pos, label, event_id, match_id) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.8))
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, line_color="black")

    panels = [("REAL defense", real_defend), ("modelA (lambda=0)", pred_a), ("modelB (lambda=0.5)", pred_b)]

    for ax, (title, defend_traj) in zip(axes, panels):
        pitch.draw(ax=ax)
        for i in range(attack_real.shape[1]):
            xs, ys = attack_real[:, i, 0], attack_real[:, i, 1]
            if (xs == 0).all() and (ys == 0).all():
                continue
            ax.plot(xs, ys, color="crimson", alpha=0.3, linewidth=1.0, zorder=1)
            ax.scatter(xs[-1], ys[-1], color="crimson", s=30, zorder=2, marker="^", alpha=0.6)

        for i in range(defend_traj.shape[1]):
            if not defend_mask[i]:
                continue
            xs, ys = defend_traj[:, i, 0], defend_traj[:, i, 1]
            ax.plot(xs, ys, color="royalblue", alpha=0.7, linewidth=1.5, zorder=3)
            ax.scatter(xs[0], ys[0], color="royalblue", s=50, zorder=4, marker="o")
            ax.scatter(xs[-1], ys[-1], color="royalblue", s=80, zorder=4, marker="^")

        ax.plot(ball_pos[:, 0], ball_pos[:, 1], color="black", linewidth=1.0, linestyle="--", zorder=5)
        ax.set_title(title, fontsize=12)

    label_str = "SUCCESS" if label == 1 else "FAILURE"
    fig.suptitle(f"{match_id} / {event_id}  result={label_str}  (red=attack(real, context), blue=defense)", fontsize=13)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"event{event_idx}_{match_id}_{event_id}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)

    ds = CounterAttackDataset(trajs)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
    target_std = compute_target_compactness(trajs, INPUT_FRAMES, side="defend")
    print(f"target_std (defend, longitudinal): {target_std:.2f}")

    print("training modelA (lambda=0)...")
    model_a = train_model(loader, lam=0.0, target_std=target_std)
    print("training modelB (lambda=0.5)...")
    model_b = train_model(loader, lam=0.5, target_std=target_std)

    # 代表イベントとして、成功2件・失敗2件を異なる試合から選ぶ
    labels = [t.label for t in trajs]
    success_idx = [i for i, l in enumerate(labels) if l == 1]
    failure_idx = [i for i, l in enumerate(labels) if l == 0]
    selected = [success_idx[0], success_idx[len(success_idx) // 2], failure_idx[0], failure_idx[len(failure_idx) // 2]]

    eval_loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_samples)
    batches = list(eval_loader)

    with torch.no_grad():
        for idx in selected:
            batch = batches[idx]
            pos_a, _, _ = model_a(batch)
            pos_b, _, _ = model_b(batch)

            real_defend = batch["target_defend_pos"][0].numpy()  # (T, n, 2)
            pred_a = pos_a[0].permute(1, 0, 2).numpy()  # (n,T,2)->(T,n,2)
            pred_b = pos_b[0].permute(1, 0, 2).numpy()
            attack_real = batch["target_attack_pos"][0].numpy()
            defend_mask = batch["defend_mask"][0].numpy()
            ball_pos = batch["input_ball_pos"][0, :, 0, :].numpy()  # 観測窓のボール位置(参考)

            path = plot_comparison(
                idx, real_defend, pred_a, pred_b, attack_real, defend_mask, ball_pos,
                batch["label"][0].item(), batch["event_id"][0], batch["match_id"][0],
            )
            print(f"saved {path}")


if __name__ == "__main__":
    main()
