"""擬似的な反実仮想シミュレーション：守備をPINNゴーストに差し替えたときの危険度比較（計画書12.2節）。

実際に成功（label=1、失点相当の代理指標）したカウンター局面について、守備側だけを
modelB（PINNゴースト、λ=0.5、運用上の主指標と同じ設定）の予測に差し替え、
攻撃側・ボールは実観測のまま固定する。この2パターン（実際の守備／PINNゴースト）
それぞれについて、`space_control.py`のピッチコントロール（守備配置に反応する指標。
xTはボール位置のみに基づくため使えない、12.2節参照）で「攻撃側がどれだけ
ピッチを支配していたか」を計算し、比較する。

【限界（12.3節・12.5節、必ず併記する）】
- 攻撃側の行動は実観測のまま固定しており、実際には守備配置の変化に攻撃側が
  反応した可能性がある（SUTVA違反）
- 反実仮想を評価する指標（ピッチコントロール）自体が正しいという、別の前提に
  依存する二重の近似
- modelBのゴースト軌道が「個別に正しい」ことを直接検証する手段はない
  （因果推論の根本問題、12.5節）
- 3者比較・コーチング可視化と同様、事例によって差の大きさにばらつきが出る

事例選定：label=1（成功）かつボールが実質的に動いている（デッドボール除外、
three_way_comparison.pyと同じ基準）事例から、実際の守備コンパクトネスが
target_stdから乖離している事例を優先し、試合の多様性を確保して選ぶ。

実行: uv run python scripts/counterfactual_pitch_control.py
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
from hs_pinn.space_control import build_pitch_grid, isotropic_gaussian_dominance
from hs_pinn.tactic_metrics import longitudinal_variance

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
OUT_DIR = Path(__file__).resolve().parent.parent / "documents" / "counterfactual_images"
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
PITCH_BOUNDS = PitchBounds(0.0, PITCH_LENGTH, 0.0, PITCH_WIDTH)
LAMBDA_OPERATIONAL = 0.5  # 運用上の主指標(11節)と同じ設定
EPOCHS = 30
BATCH_SIZE = 16
N_EVENTS = 4
MIN_BALL_PATH_LENGTH = 5.0  # three_way_comparison.pyと同じデッドボール除外基準
FRAME_STRIDE = 10  # soft_constraints.pyのspace_control_loss訓練項と同じ間引き幅
GRID = build_pitch_grid(PITCH_LENGTH, PITCH_WIDTH, nx=22, ny=15)


def ball_path_length(traj) -> float:
    ball = traj.ball_pos[INPUT_FRAMES:]
    diffs = np.linalg.norm(np.diff(ball, axis=0), axis=1)
    return float(np.nansum(diffs))


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


def danger_series(attack_pos: np.ndarray, attack_mask: np.ndarray, defend_pos: np.ndarray,
                   defend_mask: np.ndarray, frame_stride: int) -> np.ndarray:
    """attack_pos: (T, n_attack, 2), defend_pos: (n_defend, T, 2)。戻り値: (T // stride,)"""
    attack_valid = attack_pos[:, attack_mask, :]  # (T, n_attack_valid, 2)
    defend_valid = defend_pos[defend_mask]  # (n_defend_valid, T, 2)
    T = attack_pos.shape[0]
    values = []
    for t in range(0, T, frame_stride):
        values.append(isotropic_gaussian_dominance(GRID, attack_valid[t], defend_valid[:, t]))
    return np.array(values)


def plot_event(match_id: str, event_id: str, real_defend, pred_ghost, defend_mask,
               attack_real, attack_mask, ball_pos) -> tuple[Path, dict]:
    danger_real = danger_series(attack_real, attack_mask, real_defend, defend_mask, FRAME_STRIDE)
    danger_ghost = danger_series(attack_real, attack_mask, pred_ghost, defend_mask, FRAME_STRIDE)
    t_axis = np.arange(len(danger_real)) * FRAME_STRIDE / 25.0

    fig, (ax_pitch, ax_line) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, line_color="black")
    pitch.draw(ax=ax_pitch)

    for i in range(attack_real.shape[1]):
        if not attack_mask[i]:
            continue
        xs, ys = attack_real[:, i, 0], attack_real[:, i, 1]
        ax_pitch.plot(xs, ys, color="crimson", alpha=0.75, linewidth=2.0, zorder=1,
                      label="attack (real, context)" if i == 0 else None)
        ax_pitch.scatter(xs[-1], ys[-1], color="crimson", s=55, zorder=2, marker="^", alpha=0.9)

    n_defend = real_defend.shape[0]
    first_real, first_ghost = True, True
    for i in range(n_defend):
        if not defend_mask[i]:
            continue
        xs_r, ys_r = real_defend[i, :, 0], real_defend[i, :, 1]
        ax_pitch.plot(xs_r, ys_r, color="royalblue", linewidth=1.8, linestyle="-", zorder=4,
                      label="REAL defense" if first_real else None)
        first_real = False
        ax_pitch.scatter(xs_r[0], ys_r[0], color="royalblue", s=45, zorder=5, marker="o")
        ax_pitch.scatter(xs_r[-1], ys_r[-1], color="royalblue", s=75, zorder=5, marker="^")

        xs_g, ys_g = pred_ghost[i, :, 0], pred_ghost[i, :, 1]
        ax_pitch.plot(xs_g, ys_g, color="darkorange", linewidth=1.8, linestyle="--", zorder=4,
                      label="PINN ghost (theory-compliant)" if first_ghost else None)
        first_ghost = False
        ax_pitch.scatter(xs_g[0], ys_g[0], color="darkorange", s=45, zorder=5, marker="o", facecolors="none")
        ax_pitch.scatter(xs_g[-1], ys_g[-1], color="darkorange", s=75, zorder=5, marker="^", facecolors="none")

    ax_pitch.plot(ball_pos[:, 0], ball_pos[:, 1], color="black", linewidth=1.8, linestyle=":", zorder=3, alpha=0.9,
                  label="ball")
    ax_pitch.scatter(ball_pos[0, 0], ball_pos[0, 1], color="black", s=40, zorder=6, marker="o", facecolors="none")
    ax_pitch.scatter(ball_pos[-1, 0], ball_pos[-1, 1], color="black", s=40, zorder=6, marker="^")
    ax_pitch.set_title(f"{match_id} / {event_id}", fontsize=11)
    ax_pitch.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=8, frameon=False)

    ax_line.plot(t_axis, danger_real, color="royalblue", marker="o", label="danger (REAL defense)")
    ax_line.plot(t_axis, danger_ghost, color="darkorange", marker="o", linestyle="--", label="danger (PINN ghost)")
    ax_line.set_xlabel("time since ball recovery (s)")
    ax_line.set_ylabel("attack pitch-control dominance (higher = more dangerous)")
    ax_line.set_title("danger over time: REAL vs counterfactual")
    ax_line.legend(fontsize=9)
    ax_line.grid(alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"counterfactual_{match_id}_{event_id}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "match_id": match_id, "event_id": event_id,
        "danger_real_mean": float(danger_real.mean()), "danger_ghost_mean": float(danger_ghost.mean()),
        "delta": float(danger_ghost.mean() - danger_real.mean()),
    }
    return out_path, summary


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        all_trajs = pickle.load(f)

    target_std_all = compute_target_compactness(all_trajs, INPUT_FRAMES, side="defend")

    candidates = []
    for t in all_trajs:
        if t.label != 1:
            continue
        if ball_path_length(t) < MIN_BALL_PATH_LENGTH:
            continue
        window = t.defend_pos[INPUT_FRAMES:]
        stds = [np.sqrt(v) for v in (longitudinal_variance(f) for f in window) if not np.isnan(v)]
        if not stds:
            continue
        real_std = float(np.mean(stds))
        candidates.append((abs(real_std - target_std_all), t.match_id, t.event_id))

    candidates.sort(reverse=True)
    selected, seen_matches = [], set()
    for _, match_id, event_id in candidates:
        if match_id in seen_matches:
            continue
        selected.append((match_id, event_id))
        seen_matches.add(match_id)
        if len(selected) >= N_EVENTS:
            break
    print(f"selected events (label=1, non-dead-ball, high compactness deviation): {selected}")

    traj_by_key = {(t.match_id, t.event_id): t for t in all_trajs}
    matches_needed = sorted({m for m, _ in selected})
    summaries = []

    for holdout_match in matches_needed:
        train_trajs = [t for t in all_trajs if t.match_id != holdout_match]
        holdout_trajs = [t for t in all_trajs if t.match_id == holdout_match]

        train_ds = CounterAttackDataset(train_trajs)
        holdout_ds = CounterAttackDataset(holdout_trajs)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
        holdout_loader = DataLoader(holdout_ds, batch_size=1, shuffle=False, collate_fn=collate_samples)

        target_std = compute_target_compactness(train_trajs, INPUT_FRAMES, side="defend")
        print(f"[holdout={holdout_match}] target_std={target_std:.2f}  training modelB(lambda={LAMBDA_OPERATIONAL})...")
        model_b = train_model(train_loader, lam=LAMBDA_OPERATIONAL, target_std=target_std)

        for batch in holdout_loader:
            key = (batch["match_id"][0], batch["event_id"][0])
            if key not in selected:
                continue
            with torch.no_grad():
                pos_b, _, _ = model_b(batch)
            defend_mask = batch["defend_mask"][0].numpy()
            attack_mask = batch["attack_mask"][0].numpy()
            real_defend = batch["target_defend_pos"].permute(0, 2, 1, 3)[0].numpy()
            pred_ghost = pos_b[0].numpy()
            attack_real = batch["target_attack_pos"][0].numpy()
            ball_pos = traj_by_key[key].ball_pos[INPUT_FRAMES:]

            path, summary = plot_event(key[0], key[1], real_defend, pred_ghost, defend_mask,
                                        attack_real, attack_mask, ball_pos)
            summaries.append(summary)
            print(f"  saved {path}  danger_real={summary['danger_real_mean']:.4f}  "
                  f"danger_ghost={summary['danger_ghost_mean']:.4f}  delta={summary['delta']:+.4f}")

    print("\n=== summary ===")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
