"""ハード制約の価値を「精度」ではなく「物理的妥当性の保証」として検証する。

制約なしモデル（`UnconstrainedIntegrationLayer`）は、実データ（選手は常に
ピッチ内にいる）で学習したにもかかわらず、予測では物理的にありえない出力
（ピッチ外への配置、最大速度・加速度の超過）を実際に出しうる。
ハード制約モデルは構造上これらを原理的に出せない（forward passの数式が保証する）。
この非対称性を定量化する。

実行: uv run python scripts/check_physical_plausibility.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hs_pinn.dataset import CounterAttackDataset, collate_samples, split_trajectories
from hs_pinn.hard_constraints import HardConstraintLayer, PitchBounds, UnconstrainedIntegrationLayer
from hs_pinn.model import TrajectoryBackbone

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "checkpoints"

V_MAX, A_MAX = 9.0, 6.0
PITCH = PitchBounds(0.0, 105.0, 0.0, 68.0)


def load_model(ckpt_path: Path, constraint_layer: torch.nn.Module) -> TrajectoryBackbone:
    model = TrajectoryBackbone(constraint_layer=constraint_layer)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def analyze(model: TrajectoryBackbone, batch: dict, label: str) -> None:
    with torch.no_grad():
        pos, vel, acc = model(batch)  # (B, n_attack, T, 2)
    mask = batch["attack_mask"]  # (B, n_attack)
    frame_mask = mask.unsqueeze(-1).expand(-1, -1, pos.shape[-2])  # (B, n_attack, T)

    speed = vel.norm(dim=-1)
    accel_mag = acc.norm(dim=-1)
    x, y = pos[..., 0], pos[..., 1]
    out_of_pitch = (x < -1e-6) | (x > PITCH.x_max + 1e-6) | (y < -1e-6) | (y > PITCH.y_max + 1e-6)

    n_total = frame_mask.sum().item()
    n_speed = (speed[frame_mask] > V_MAX + 1e-6).sum().item()
    n_accel = (accel_mag[frame_mask] > A_MAX + 1e-6).sum().item()
    n_pitch = out_of_pitch[frame_mask].sum().item()

    # 軌道単位（選手×イベント）で「1フレームでも違反したか」を集計
    traj_violation = (out_of_pitch & frame_mask).any(dim=-1)  # (B, n_attack)
    n_traj = mask.sum().item()
    n_traj_violation = traj_violation[mask].sum().item()

    print(f"=== {label} ===")
    print(f"  n_frames={n_total}")
    print(f"  speed>{V_MAX}m/s: {n_speed} ({100*n_speed/n_total:.3f}%)  max={speed[frame_mask].max().item():.2f}m/s")
    print(f"  accel>{A_MAX}m/s2: {n_accel} ({100*n_accel/n_total:.3f}%)  max={accel_mag[frame_mask].max().item():.2f}m/s2")
    print(f"  frame outside pitch: {n_pitch} ({100*n_pitch/n_total:.3f}%)")
    print(f"  trajectories with >=1 out-of-pitch frame: {n_traj_violation}/{n_traj} ({100*n_traj_violation/n_traj:.2f}%)")
    print(f"  x range=({x[frame_mask].min().item():.1f}, {x[frame_mask].max().item():.1f})  "
          f"y range=({y[frame_mask].min().item():.1f}, {y[frame_mask].max().item():.1f})")
    print()


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)
    splits = split_trajectories(trajs)

    valid_ds = CounterAttackDataset(splits["valid"])
    train_ds = CounterAttackDataset(splits["train"])
    batch_valid = next(iter(DataLoader(valid_ds, batch_size=len(valid_ds), shuffle=False, collate_fn=collate_samples)))
    batch_train = next(iter(DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, collate_fn=collate_samples)))

    m_none = load_model(CHECKPOINT_DIR / "none_100ep" / "epoch_100.pt", UnconstrainedIntegrationLayer(dt=1 / 25))
    m_hard = load_model(
        CHECKPOINT_DIR / "hard_only_100ep" / "epoch_100.pt",
        HardConstraintLayer(a_max=A_MAX, v_max=V_MAX, dt=1 / 25, pitch_bounds=PITCH),
    )

    analyze(m_none, batch_train, "制約なし（train）")
    analyze(m_none, batch_valid, "制約なし（valid）")
    analyze(m_hard, batch_train, "ハードのみ（train、sanity check）")
    analyze(m_hard, batch_valid, "ハードのみ（valid、sanity check）")


if __name__ == "__main__":
    main()
