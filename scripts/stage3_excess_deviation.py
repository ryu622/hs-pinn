"""段階3：Leave-One-Match-OutでmodelA/Bを構築し、超過逸脱度を計算する（計画書10.4節・10.6節）。

7試合それぞれを1回ずつholdoutし、残り6試合でmodelA（λ=0、データ駆動ghost）と
modelB（λ=2.0、理論駆動ghost。段階2のスクリーニングで(a)(b)(c)を満たした設定）を学習、
holdout試合の全イベントについてheld-out予測を得る。これを7回繰り返し、502件全件の
held-out予測（modelA・modelBとも）を収集する。

超過逸脱度 = dist(modelB, 実観測) − dist(modelA, 実観測)

実行: uv run python scripts/stage3_excess_deviation.py
"""

from __future__ import annotations

import argparse
import pickle
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
PITCH_BOUNDS = PitchBounds(0.0, 105.0, 0.0, 68.0)

ALL_MATCHES = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]
EPOCHS = 30
BATCH_SIZE = 16


def per_event_distance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred, target: (B, n_defend, T, 2), mask: (B, n_defend) -> (B,) イベントごとの平均距離"""
    diff = (pred - target).norm(dim=-1)  # (B, n_defend, T)
    m = mask.unsqueeze(-1).expand_as(diff).float()
    return (diff * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0)


def train_model(train_loader, lam: float, target_std: float, seed: int = 0) -> TrajectoryBackbone:
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
            loss = L_data
            if lam > 0:
                L_compact = compactness_loss(pos, mask, target_std)
                loss = loss + lam * L_compact

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def evaluate(model: TrajectoryBackbone, loader) -> list[dict]:
    records = []
    with torch.no_grad():
        for batch in loader:
            pos, _, _ = model(batch)
            target = batch["target_defend_pos"].permute(0, 2, 1, 3)
            mask = batch["defend_mask"]
            dist = per_event_distance(pos, target, mask)
            for i in range(len(batch["match_id"])):
                records.append(
                    {
                        "match_id": batch["match_id"][i],
                        "event_id": batch["event_id"][i],
                        "label": int(batch["label"][i]),
                        "recovery_x": batch["input_ball_pos"][i, 0, 0, 0].item(),
                        "dist": dist[i].item(),
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_b", type=float, default=2.0)
    args = parser.parse_args()
    lambda_b = args.lambda_b
    out_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "processed" / f"stage3_excess_deviation_lam{lambda_b}.pkl"
    )

    with open(CACHE_PATH, "rb") as f:
        all_trajs = pickle.load(f)

    records_a, records_b = [], []

    for holdout_match in ALL_MATCHES:
        train_trajs = [t for t in all_trajs if t.match_id != holdout_match]
        holdout_trajs = [t for t in all_trajs if t.match_id == holdout_match]

        train_ds = CounterAttackDataset(train_trajs)
        holdout_ds = CounterAttackDataset(holdout_trajs)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
        holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_samples)

        target_std = compute_target_compactness(train_trajs, INPUT_FRAMES, side="defend")

        model_a = train_model(train_loader, lam=0.0, target_std=target_std, seed=0)
        model_b = train_model(train_loader, lam=lambda_b, target_std=target_std, seed=0)

        recs_a = evaluate(model_a, holdout_loader)
        recs_b = evaluate(model_b, holdout_loader)
        records_a.extend(recs_a)
        records_b.extend(recs_b)

        print(f"{holdout_match}: n_holdout={len(holdout_trajs)}  "
              f"modelA_mean_dist={sum(r['dist'] for r in recs_a)/len(recs_a):.3f}  "
              f"modelB_mean_dist={sum(r['dist'] for r in recs_b)/len(recs_b):.3f}")

    # イベント単位でmodelA/Bの結果を突き合わせ、超過逸脱度を計算
    by_key_a = {(r["match_id"], r["event_id"]): r for r in records_a}
    by_key_b = {(r["match_id"], r["event_id"]): r for r in records_b}

    combined = []
    for key, ra in by_key_a.items():
        rb = by_key_b[key]
        combined.append(
            {
                "match_id": ra["match_id"],
                "event_id": ra["event_id"],
                "label": ra["label"],
                "recovery_x": ra["recovery_x"],
                "dist_A": ra["dist"],
                "dist_B": rb["dist"],
                "excess_deviation": rb["dist"] - ra["dist"],
            }
        )

    print(f"\ntotal events: {len(combined)}")
    with open(out_path, "wb") as f:
        pickle.dump(combined, f)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
