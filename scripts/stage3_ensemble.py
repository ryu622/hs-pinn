"""アンサンブルによる超過逸脱度のノイズ低減（前回の議論に対応）。

超過逸脱度 = dist(modelB,実観測) - dist(modelA,実観測) は、2つの独立に
学習したモデルの予測誤差の差分であるため、単一シードでは学習自体の
ランダムなばらつきが乗る。各fold・各モデル種別でN_SEEDS個学習し、
「予測位置」を平均してから距離を計算する（距離を計算してから平均する
より、先に予測を平均する方が分散削減の効果が直接的）。

既定はλ=0.5（複数シード確認で最も頑健だった設定、documents/stage3_multiseed_report.md）。
--lambda_bで他の値も試せる（例：λ=4, 8がアンサンブルで安定するかの診断）。

実行: uv run python scripts/stage3_ensemble.py [--lambda_b 8.0]
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
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
N_SEEDS = 5
EPOCHS = 30
BATCH_SIZE = 16


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


def get_predictions(model: TrajectoryBackbone, loader) -> dict[tuple[str, str], np.ndarray]:
    """key -> (n_defend, T, 2) 予測位置（選手軸→時間軸の順、real側と合わせる）"""
    out = {}
    with torch.no_grad():
        for batch in loader:
            pos, _, _ = model(batch)  # (B, n_defend, T, 2)
            for i in range(len(batch["match_id"])):
                key = (batch["match_id"][i], batch["event_id"][i])
                out[key] = pos[i].numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_b", type=float, default=0.5)
    args = parser.parse_args()
    lambda_b = args.lambda_b
    out_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "processed" / f"stage3_ensemble_lam{lambda_b}.pkl"
    )

    with open(CACHE_PATH, "rb") as f:
        all_trajs = pickle.load(f)

    records = []
    t_start = time.time()

    for holdout_match in ALL_MATCHES:
        train_trajs = [t for t in all_trajs if t.match_id != holdout_match]
        holdout_trajs = [t for t in all_trajs if t.match_id == holdout_match]

        train_ds = CounterAttackDataset(train_trajs)
        holdout_ds = CounterAttackDataset(holdout_trajs)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
        holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_samples)

        target_std = compute_target_compactness(train_trajs, INPUT_FRAMES, side="defend")

        preds_a_by_seed, preds_b_by_seed = [], []
        for seed in range(N_SEEDS):
            model_a = train_model(train_loader, lam=0.0, target_std=target_std, seed=seed)
            preds_a_by_seed.append(get_predictions(model_a, holdout_loader))
            model_b = train_model(train_loader, lam=lambda_b, target_std=target_std, seed=seed)
            preds_b_by_seed.append(get_predictions(model_b, holdout_loader))

        # real位置・mask・ラベル・奪取位置を1回だけ収集
        real, masks, labels, recovery_x = {}, {}, {}, {}
        for batch in holdout_loader:
            target = batch["target_defend_pos"].permute(0, 2, 1, 3)  # (B, n_defend, T, 2)
            for i in range(len(batch["match_id"])):
                key = (batch["match_id"][i], batch["event_id"][i])
                real[key] = target[i].numpy()
                masks[key] = batch["defend_mask"][i].numpy()
                labels[key] = int(batch["label"][i])
                recovery_x[key] = batch["input_ball_pos"][i, 0, 0, 0].item()

        for key in real:
            avg_a = np.mean([preds_a_by_seed[s][key] for s in range(N_SEEDS)], axis=0)
            avg_b = np.mean([preds_b_by_seed[s][key] for s in range(N_SEEDS)], axis=0)
            m = masks[key]

            diff_a = np.linalg.norm(avg_a - real[key], axis=-1)  # (n_defend, T)
            diff_b = np.linalg.norm(avg_b - real[key], axis=-1)
            dist_a = diff_a[m].mean()
            dist_b = diff_b[m].mean()

            records.append(
                {
                    "match_id": key[0],
                    "event_id": key[1],
                    "label": labels[key],
                    "recovery_x": recovery_x[key],
                    "dist_A": float(dist_a),
                    "dist_B": float(dist_b),
                    "excess_deviation": float(dist_b - dist_a),
                }
            )

        elapsed = time.time() - t_start
        print(f"holdout={holdout_match} done ({elapsed:.0f}s elapsed)")

    print(f"\ntotal events: {len(records)}")
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
