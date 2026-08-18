"""λの効果と学習ノイズを分離するための複数シード実験（計画書10.10節）。

`documents/stage3_lambda_sweep_report.md`で1シードのみのλスイープが
滑らかでない谷（λ=0.5→1→2）を示したため、乱数シードを複数（0,1,2）で
繰り返し、λごとの分散を見積もる。modelA（λ非依存）はseed・fold単位で
1回だけ学習し、λごとに使い回す（学習コスト削減）。

実行: uv run python scripts/stage3_multiseed_sweep.py
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import torch
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
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "stage3_multiseed.pkl"
PITCH_BOUNDS = PitchBounds(0.0, 105.0, 0.0, 68.0)

ALL_MATCHES = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]
SEEDS = [0, 1, 2]
LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
EPOCHS = 30
BATCH_SIZE = 16


def per_event_distance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).norm(dim=-1)
    m = mask.unsqueeze(-1).expand_as(diff).float()
    return (diff * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0)


def train_model(train_loader, lam: float, target_std: float, seed: int) -> TrajectoryBackbone:
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
            loss = diff[mask].mean()
            if lam > 0:
                loss = loss + lam * compactness_loss(pos, mask, target_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def evaluate(model: TrajectoryBackbone, loader) -> dict[tuple[str, str], float]:
    out = {}
    with torch.no_grad():
        for batch in loader:
            pos, _, _ = model(batch)
            target = batch["target_defend_pos"].permute(0, 2, 1, 3)
            mask = batch["defend_mask"]
            dist = per_event_distance(pos, target, mask)
            for i in range(len(batch["match_id"])):
                out[(batch["match_id"][i], batch["event_id"][i])] = dist[i].item()
    return out


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        all_trajs = pickle.load(f)

    records = []
    t_start = time.time()

    for seed in SEEDS:
        for holdout_match in ALL_MATCHES:
            train_trajs = [t for t in all_trajs if t.match_id != holdout_match]
            holdout_trajs = [t for t in all_trajs if t.match_id == holdout_match]

            train_ds = CounterAttackDataset(train_trajs)
            holdout_ds = CounterAttackDataset(holdout_trajs)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
            holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_samples)

            target_std = compute_target_compactness(train_trajs, INPUT_FRAMES, side="defend")

            model_a = train_model(train_loader, lam=0.0, target_std=target_std, seed=seed)
            dist_a = evaluate(model_a, holdout_loader)

            # 奪取位置・ラベルはholdout_loaderを1回舐めて取得
            recovery_x = {}
            labels = {}
            for batch in holdout_loader:
                for i in range(len(batch["match_id"])):
                    key = (batch["match_id"][i], batch["event_id"][i])
                    recovery_x[key] = batch["input_ball_pos"][i, 0, 0, 0].item()
                    labels[key] = int(batch["label"][i])

            for lam in LAMBDAS:
                model_b = train_model(train_loader, lam=lam, target_std=target_std, seed=seed)
                dist_b = evaluate(model_b, holdout_loader)

                for key in dist_a:
                    records.append(
                        {
                            "seed": seed,
                            "lambda": lam,
                            "match_id": key[0],
                            "event_id": key[1],
                            "label": labels[key],
                            "recovery_x": recovery_x[key],
                            "dist_A": dist_a[key],
                            "dist_B": dist_b[key],
                            "excess_deviation": dist_b[key] - dist_a[key],
                        }
                    )

            elapsed = time.time() - t_start
            print(f"seed={seed} holdout={holdout_match} done ({elapsed:.0f}s elapsed)")

    print(f"\ntotal records: {len(records)}")
    with open(OUT_PATH, "wb") as f:
        pickle.dump(records, f)
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
