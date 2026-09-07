"""反実仮想シミュレーションの全件版（計画書12.6節の4事例例示を全件に拡張）。

`counterfactual_pitch_control.py`の4事例例示と同じ「守備をPINNゴーストに
差し替え、space_control.pyのピッチコントロールで危険度を比較する」手順を、
label=1（成功、失点相当の代理指標）の全事例・および全502件に対して機械的に
実行し、「危険度が下がった事例が何%あったか」「平均でどれだけ下がったか」を
統計的に集計する。

7試合すべてをホールドアウトしてmodelB(λ=0.5)をLOMO-CVで学習する。既存の
counterfactual_pitch_control.pyから学習・危険度計算のロジックを再利用する。

実行: uv run python scripts/counterfactual_full_sweep.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from counterfactual_pitch_control import (
    CACHE_PATH,
    LAMBDA_OPERATIONAL,
    BATCH_SIZE,
    MIN_BALL_PATH_LENGTH,
    FRAME_STRIDE,
    ball_path_length,
    train_model,
    danger_series,
)
from hs_pinn.dataset import INPUT_FRAMES, CounterAttackDataset, collate_samples
from hs_pinn.soft_constraints import compute_target_compactness

ALL_MATCHES = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]
OUT_PKL = Path(__file__).resolve().parent.parent / "data" / "processed" / "counterfactual_full_sweep.pkl"
OUT_FIG = Path(__file__).resolve().parent.parent / "documents" / "counterfactual_full_sweep_histogram.png"


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        all_trajs = pickle.load(f)

    records: list[dict] = []
    for holdout_match in ALL_MATCHES:
        train_trajs = [t for t in all_trajs if t.match_id != holdout_match]
        holdout_trajs = [t for t in all_trajs if t.match_id == holdout_match and ball_path_length(t) >= MIN_BALL_PATH_LENGTH]
        if not holdout_trajs:
            continue

        train_ds = CounterAttackDataset(train_trajs)
        holdout_ds = CounterAttackDataset(holdout_trajs)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
        holdout_loader = DataLoader(holdout_ds, batch_size=1, shuffle=False, collate_fn=collate_samples)

        target_std = compute_target_compactness(train_trajs, INPUT_FRAMES, side="defend")
        print(f"[holdout={holdout_match}] n_events={len(holdout_trajs)} target_std={target_std:.2f} training modelB(lambda={LAMBDA_OPERATIONAL})...")
        model_b = train_model(train_loader, lam=LAMBDA_OPERATIONAL, target_std=target_std)

        for batch in holdout_loader:
            key = (batch["match_id"][0], batch["event_id"][0])
            with torch.no_grad():
                pos_b, _, _ = model_b(batch)
            defend_mask = batch["defend_mask"][0].numpy()
            attack_mask = batch["attack_mask"][0].numpy()
            real_defend = batch["target_defend_pos"].permute(0, 2, 1, 3)[0].numpy()
            pred_ghost = pos_b[0].numpy()
            attack_real = batch["target_attack_pos"][0].numpy()
            label = int(batch["label"][0])

            danger_real = danger_series(attack_real, attack_mask, real_defend, defend_mask, FRAME_STRIDE)
            danger_ghost = danger_series(attack_real, attack_mask, pred_ghost, defend_mask, FRAME_STRIDE)

            records.append({
                "match_id": key[0], "event_id": key[1], "label": label,
                "danger_real_mean": float(danger_real.mean()),
                "danger_ghost_mean": float(danger_ghost.mean()),
                "delta": float(danger_ghost.mean() - danger_real.mean()),
            })
        print(f"  done. cumulative n={len(records)}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(records, f)
    print(f"saved {OUT_PKL}")

    def summarize(recs: list[dict], name: str) -> None:
        deltas = np.array([r["delta"] for r in recs])
        n = len(deltas)
        frac_negative = (deltas < 0).mean() * 100
        print(f"\n=== {name} (n={n}) ===")
        print(f"  delta < 0 (ゴーストの方が危険度低い) の割合: {frac_negative:.1f}%")
        print(f"  delta 平均={deltas.mean():.4f}  標準偏差={deltas.std():.4f}  中央値={np.median(deltas):.4f}")
        matches = sorted({r["match_id"] for r in recs})
        print(f"  試合数: {len(matches)}")
        for m in matches:
            sub = np.array([r["delta"] for r in recs if r["match_id"] == m])
            print(f"    {m}: n={len(sub)}  delta平均={sub.mean():.4f}  delta<0割合={(sub < 0).mean() * 100:.1f}%")

    label1 = [r for r in records if r["label"] == 1]
    summarize(label1, "label=1 (成功、失点相当の代理指標)")
    summarize(records, "全体(デッドボール除外後)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist([r["delta"] for r in label1], bins=20, color="steelblue", alpha=0.8, edgecolor="black")
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("delta = danger(PINN ghost) - danger(REAL)  (negative = ghost safer)")
    ax.set_ylabel("count")
    ax.set_title(f"distribution of counterfactual danger delta (label=1, n={len(label1)})")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
