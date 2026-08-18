"""超過逸脱度に基づく代表シーンの可視化（計画書10.7節 Part B）。

`data/processed/excess_deviation_aggregated.pkl`（λ=0.5・5シードアンサンブル版）
から「超過逸脱度の高低 × 成功/失敗」の2x2で4事例を選び、各事例についてmodelA
（λ=0）・modelB（λ=0.5）を5シードアンサンブルで再学習し、「実際の守備」
「modelA予測（平均）」「modelB予測（平均）」を並べてピッチ図にする。

選定基準:
- 成功×超過逸脱度が高い：セオリーからの逸脱が失点に結びついた、理論に最も整合する事例
- 成功×超過逸脱度が低い（マイナス）：守備はセオリー通りだったのに失点した例外
- 失敗×超過逸脱度が高い：守備はセオリーから逸脱したのに守り切れた例外
- 失敗×超過逸脱度が低い（マイナス）：セオリー通りの守備が守り切った、理論に最も整合する事例

実行: uv run python scripts/plot_excess_deviation_cases.py
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

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
AGG_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "excess_deviation_aggregated.pkl"
OUT_DIR = Path(__file__).resolve().parent.parent / "documents" / "excess_deviation_case_images"
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
PITCH_BOUNDS = PitchBounds(0.0, PITCH_LENGTH, 0.0, PITCH_WIDTH)
LAMBDA_B = 0.5
N_SEEDS = 5
EPOCHS = 30
BATCH_SIZE = 16


def select_cases(records: list[dict]) -> list[tuple[str, dict]]:
    succ = sorted([r for r in records if r["label"] == 1], key=lambda r: -r["excess_deviation"])
    fail = sorted([r for r in records if r["label"] == 0], key=lambda r: -r["excess_deviation"])
    return [
        ("success_high_excess", succ[0]),
        ("success_low_excess", succ[-1]),
        ("failure_high_excess", fail[0]),
        ("failure_low_excess", fail[-1]),
    ]


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
    out = {}
    with torch.no_grad():
        for batch in loader:
            pos, _, _ = model(batch)  # (B, n_defend, T, 2)
            for i in range(len(batch["match_id"])):
                key = (batch["match_id"][i], batch["event_id"][i])
                out[key] = pos[i].numpy()
    return out


def plot_comparison(tag: str, case: dict, real_defend, pred_a, pred_b, attack_real, defend_mask, ball_pos) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.8))
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, line_color="black")

    panels = [("REAL defense", real_defend), ("modelA (lambda=0)", pred_a), ("modelB (lambda=0.5, ensemble avg)", pred_b)]

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

    label_str = "SUCCESS" if case["label"] == 1 else "FAILURE"
    fig.suptitle(
        f"[{tag}] {case['match_id']} / {case['event_id']}  result={label_str}  "
        f"excess_deviation={case['excess_deviation']:+.3f}  defend_team={case['defend_team_id']}\n"
        f"(red=attack(real, context), blue=defense)",
        fontsize=13,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{tag}_{case['match_id']}_{case['event_id']}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        all_trajs = pickle.load(f)
    with open(AGG_PATH, "rb") as f:
        records = pickle.load(f)

    cases = select_cases(records)
    for tag, case in cases:
        print(f"{tag}: {case['match_id']} / {case['event_id']}  excess_deviation={case['excess_deviation']:+.3f}")

    needed_matches = sorted({case["match_id"] for _, case in cases})
    print(f"\nneed to train ensembles for holdout matches: {needed_matches}")

    results: dict[tuple[str, str], dict] = {}

    for holdout_match in needed_matches:
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
            model_b = train_model(train_loader, lam=LAMBDA_B, target_std=target_std, seed=seed)
            preds_b_by_seed.append(get_predictions(model_b, holdout_loader))
        print(f"holdout={holdout_match}: trained {N_SEEDS} seeds x 2 models")

        real, masks, attack_real, ball_pos = {}, {}, {}, {}
        for batch in holdout_loader:
            for i in range(len(batch["match_id"])):
                key = (batch["match_id"][i], batch["event_id"][i])
                real[key] = batch["target_defend_pos"][i].numpy()  # (T, n_defend, 2)
                masks[key] = batch["defend_mask"][i].numpy()
                attack_real[key] = batch["target_attack_pos"][i].numpy()
                ball_pos[key] = batch["input_ball_pos"][i, :, 0, :].numpy()

        for key in real:
            avg_a = np.mean([preds_a_by_seed[s][key] for s in range(N_SEEDS)], axis=0)  # (n,T,2)
            avg_b = np.mean([preds_b_by_seed[s][key] for s in range(N_SEEDS)], axis=0)
            results[key] = {
                "real_defend": real[key],
                "pred_a": avg_a.transpose(1, 0, 2),  # (n,T,2)->(T,n,2)
                "pred_b": avg_b.transpose(1, 0, 2),
                "attack_real": attack_real[key],
                "defend_mask": masks[key],
                "ball_pos": ball_pos[key],
            }

    for tag, case in cases:
        key = (case["match_id"], case["event_id"])
        r = results[key]
        path = plot_comparison(
            tag, case, r["real_defend"], r["pred_a"], r["pred_b"], r["attack_real"], r["defend_mask"], r["ball_pos"]
        )
        print(f"saved {path}")


if __name__ == "__main__":
    main()
