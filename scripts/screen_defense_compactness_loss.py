"""段階2：守備側コンパクトネスの損失関数としての機能スクリーニング（計画書10.6節）。

判定基準を事前に固定する：
  (a) 勾配が病的でない（微分可能・勾配消失/爆発なし）
  (b) λを大きくした際にmodelBがmodelAと有意に異なる出力を示す
      （＝λを上げるとvalid L_compactが実際に改善するか）
  (c) 退化解に陥らない（全選手が1点に収束する等の非現実的な出力になっていないか）

小規模データ・少エポックでλを極端に振り、目視で確認する。守備側予測モデル
（`predict_side="defend"`）を使う点、L_dataがdefend側のADE/FDEになる点以外は
`scripts/train.py`と同じ構成。

実行: uv run python scripts/screen_defense_compactness_loss.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hs_pinn.dataset import (
    INPUT_FRAMES,
    CounterAttackDataset,
    collate_samples,
    split_trajectories,
)
from hs_pinn.hard_constraints import HardConstraintLayer, PitchBounds
from hs_pinn.model import TrajectoryBackbone
from hs_pinn.soft_constraints import compactness_loss, compute_target_compactness

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
PITCH_BOUNDS = PitchBounds(0.0, 105.0, 0.0, 68.0)
EPOCHS = 30
BATCH_SIZE = 16
LAMBDAS = [0.0, 0.5, 2.0, 8.0]  # 極端な値まで含めて振る


def masked_ade(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (pred - target).norm(dim=-1)
    return diff[mask].mean().item()


def run(lam: float, train_loader, valid_loader, target_std: float, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = TrajectoryBackbone(
        constraint_layer=HardConstraintLayer(a_max=6.0, v_max=9.0, dt=1 / 25, pitch_bounds=PITCH_BOUNDS),
        predict_side="defend",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            pos, _, _ = model(batch)
            target = batch["target_defend_pos"].permute(0, 2, 1, 3)
            mask = batch["defend_mask"]

            diff = (pos - target).norm(dim=-1)
            L_data = diff[mask].mean()
            L_compact = compactness_loss(pos, mask, target_std)
            loss = L_data if lam == 0.0 else L_data + lam * L_compact

            optimizer.zero_grad()
            loss.backward()

            grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            if not (grad_norm == grad_norm) or grad_norm > 1e4:  # NaN or 爆発
                return {"status": "gradient_pathological", "grad_norm": grad_norm}

            optimizer.step()

    # valid評価
    model.eval()
    with torch.no_grad():
        total_ade, total_compact, n, min_pairwise_dist = 0.0, 0.0, 0, []
        for batch in valid_loader:
            pos, _, _ = model(batch)
            target = batch["target_defend_pos"].permute(0, 2, 1, 3)
            mask = batch["defend_mask"]
            bsz = pos.shape[0]
            total_ade += masked_ade(pos, target, mask) * bsz
            total_compact += compactness_loss(pos, mask, target_std).item() * bsz
            n += bsz

            # 退化解チェック：各サンプル最終フレームでの選手間最小距離
            last = pos[:, :, -1, :]  # (B, n_defend, 2)
            for b in range(bsz):
                valid_players = last[b][mask[b]]
                if valid_players.shape[0] >= 2:
                    d = torch.cdist(valid_players, valid_players)
                    d.fill_diagonal_(float("inf"))
                    min_pairwise_dist.append(d.min().item())

    return {
        "status": "ok",
        "valid_ADE": total_ade / n,
        "valid_L_compact": total_compact / n,
        "min_pairwise_dist_mean": sum(min_pairwise_dist) / len(min_pairwise_dist),
        "min_pairwise_dist_worst": min(min_pairwise_dist),
    }


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)
    splits = split_trajectories(trajs)

    train_ds = CounterAttackDataset(splits["train"])
    valid_ds = CounterAttackDataset(splits["valid"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_samples)

    target_std = compute_target_compactness(splits["train"], INPUT_FRAMES, side="defend")
    print(f"defend target_std (longitudinal): {target_std:.2f}\n")

    print(f"{'lambda':>8s} {'status':>14s} {'valid_ADE':>10s} {'valid_Lcompact':>15s} "
          f"{'min_dist_mean':>14s} {'min_dist_worst':>15s}")
    for lam in LAMBDAS:
        result = run(lam, train_loader, valid_loader, target_std)
        if result["status"] != "ok":
            print(f"{lam:8g} {result['status']:>14s}")
            continue
        print(
            f"{lam:8g} {'ok':>14s} {result['valid_ADE']:10.3f} {result['valid_L_compact']:15.3f} "
            f"{result['min_pairwise_dist_mean']:14.2f} {result['min_pairwise_dist_worst']:15.2f}"
        )


if __name__ == "__main__":
    main()
