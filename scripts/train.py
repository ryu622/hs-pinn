"""最小構成での学習ループ（計画書6.1節：制約なし／ハードのみ／ハード+ソフトの3パターン比較用）。

実行例:
  uv run python scripts/train.py --constraint_mode hard_only --epochs 20
  uv run python scripts/train.py --constraint_mode hard_soft --lam 0.1 --epochs 20

ローカル(M5)での動作確認・デバッグ用の最小構成（計画書4.4節）。
本番の複数λ・複数パターン一括実験はColabで実行する想定。

【既知の制約】`hard_soft`モードの学習損失にはL_compactのみを組み込み、L_spaceは
含めていない（監視用には引き続き毎エポック計算・ログする）。near_radius絞り込み・
w2を20倍にする等の対策を試したが、バッチごとに実際の守備側配置が異なるため
L_spaceの勾配方向がバッチ間で一貫せず、学習損失としては機能しなかったため
（詳細な経緯はdocuments参照）。
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hs_pinn.dataset import (
    INPUT_FRAMES,
    CounterAttackDataset,
    collate_samples,
    split_trajectories,
)
from hs_pinn.hard_constraints import HardConstraintLayer, PitchBounds, UnconstrainedIntegrationLayer
from hs_pinn.model import TrajectoryBackbone
from hs_pinn.soft_constraints import compactness_loss, compute_target_compactness, space_control_loss
from hs_pinn.space_control import build_pitch_grid

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "checkpoints"
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0


def get_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def masked_ade_fde(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """pred, target: (B, n_attack, T, 2), mask: (B, n_attack)"""
    diff = (pred - target).norm(dim=-1)  # (B, n_attack, T)
    ade = diff[mask].mean().item()
    fde = diff[..., -1][mask].mean().item()
    return ade, fde


def run_epoch(
    model: TrajectoryBackbone,
    loader: DataLoader,
    grid: torch.Tensor,
    target_std: float,
    lam: float,
    w1: float,
    w2: float,  # 現在は学習損失には未使用（L_spaceは監視専用のため）。ログ記録用に残している
    use_soft: bool,
    optimizer: torch.optim.Optimizer | None,
    near_radius: float = 15.0,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total = {"L_data": 0.0, "L_compact": 0.0, "L_space": 0.0, "ADE": 0.0, "FDE": 0.0, "n": 0}

    for batch in loader:
        with torch.set_grad_enabled(is_train):
            pos, _, _ = model(batch)
            target = batch["target_attack_pos"].permute(0, 2, 1, 3)
            target_defend = batch["target_defend_pos"].permute(0, 2, 1, 3)
            mask = batch["attack_mask"]
            defend_mask = batch["defend_mask"]

            diff = (pos - target).norm(dim=-1)
            L_data = diff[mask].mean()

            # L_compactは学習損失に使う場合のみ勾配を通す。L_spaceは監視専用
            # （バッチ間で勾配方向が安定せず学習損失としては機能しなかったため、
            # 常にdetachしたposに対して計算する。モジュールdocstring参照）。
            L_compact = compactness_loss(pos.detach() if not use_soft else pos, mask, target_std)
            L_space = space_control_loss(
                pos.detach(),
                target_defend,
                mask,
                defend_mask,
                grid,
                near_radius=near_radius,
            )

            loss = L_data
            if use_soft:
                loss = loss + lam * w1 * L_compact

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ade, fde = masked_ade_fde(pos.detach(), target, mask)
        bsz = pos.shape[0]
        total["L_data"] += L_data.item() * bsz
        total["L_compact"] += L_compact.item() * bsz
        total["L_space"] += L_space.item() * bsz
        total["ADE"] += ade * bsz
        total["FDE"] += fde * bsz
        total["n"] += bsz

    n = total.pop("n")
    return {k: v / n for k, v in total.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraint_mode", choices=["none", "hard_only", "hard_soft"], required=True)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--near_radius", type=float, default=15.0)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.run_name or f"{args.constraint_mode}_lam{args.lam}_{int(time.time())}"
    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"run: {run_name}  device: {device}")

    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)
    splits = split_trajectories(trajs)

    train_ds = CounterAttackDataset(splits["train"])
    valid_ds = CounterAttackDataset(splits["valid"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_samples)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)

    target_std = compute_target_compactness(splits["train"], INPUT_FRAMES)
    grid = torch.from_numpy(build_pitch_grid(PITCH_LENGTH, PITCH_WIDTH)).float()

    pitch_bounds = PitchBounds(0, PITCH_LENGTH, 0, PITCH_WIDTH)
    if args.constraint_mode == "none":
        constraint_layer = UnconstrainedIntegrationLayer(dt=1 / 25)
    else:
        constraint_layer = HardConstraintLayer(a_max=6.0, v_max=9.0, dt=1 / 25, pitch_bounds=pitch_bounds)
    use_soft = args.constraint_mode == "hard_soft"

    model = TrajectoryBackbone(constraint_layer=constraint_layer)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = {
        "run_name": run_name,
        "git_commit": get_git_commit(),
        "constraint_mode": args.constraint_mode,
        "lam": args.lam,
        "w1": args.w1,
        "w2": args.w2,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "target_compactness": target_std,
        "near_radius": args.near_radius,
        "n_train": len(train_ds),
        "n_valid": len(valid_ds),
        "history": [],
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model, train_loader, grid, target_std, args.lam, args.w1, args.w2, use_soft, optimizer, args.near_radius
        )
        valid_metrics = run_epoch(
            model, valid_loader, grid, target_std, args.lam, args.w1, args.w2, use_soft, None, args.near_radius
        )
        elapsed = time.time() - t0

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train ADE={train_metrics['ADE']:.3f} FDE={train_metrics['FDE']:.3f} "
            f"L_compact={train_metrics['L_compact']:.2f} L_space={train_metrics['L_space']:.3f}  "
            f"valid ADE={valid_metrics['ADE']:.3f} FDE={valid_metrics['FDE']:.3f}  "
            f"({elapsed:.1f}s)"
        )

        log["history"].append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            ckpt_path = run_dir / f"epoch_{epoch}.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch},
                ckpt_path,
            )
            with open(run_dir / "log.json", "w") as f:
                json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\nsaved run to {run_dir}")


if __name__ == "__main__":
    main()
