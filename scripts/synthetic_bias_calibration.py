"""課題B：合成データによる構造的バイアスのキャリブレーション（計画書10.11節・10.10節項目14）。

超過逸脱度＝d(観測,modelB)-d(観測,modelA) は、modelBがL_theoryにより単一の
「理想形」に強く引き寄せられる分、たとえ真の戦術逸脱がゼロでも構造的に
modelAより観測から離れやすい（バイアス・バリアンス由来の「構造的な下駄」）
可能性がある。

これを測るため、「理論（コンパクトネスへの矯正）が完全に正しい」と仮定した
疑似観測を作る：
  1. 全データでmodelB(λ=0.5)を1回学習し、その予測をY_theory（理論的に
     理想的な守備）とする
  2. Y_theoryに現実的なノイズを加えてY_synth（疑似「観測」）を作る。
     真の戦術逸脱はゼロ（Y_synthはY_theoryにノイズを足しただけ）
  3. 実データと全く同じ7-fold LOMO CVパイプラインで、Y_synthを新しい
     「観測」としてmodelA_synth（λ=0）・modelB_synth（λ=0.5）を学習し、
     超過逸脱度を計算する
  4. 真の逸脱がゼロのはずなのに出てくる超過逸脱度が、構造的な下駄の大きさ

ノイズの大きさ(sigma)は、まずJ03WPY 1foldのみでいくつか試し、
modelA_synthのdist_A_synthが実データのdist_A(平均約2.43m)に近くなる値を選ぶ
（＝疑似データの「予測しにくさ」を実データと揃えることで、公平な比較にする）。

実行: uv run python scripts/synthetic_bias_calibration.py
"""

from __future__ import annotations

import pickle
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
EPOCHS = 30
BATCH_SIZE = 16
LAMBDA_B = 0.5
SEED = 0
REAL_DIST_A_REFERENCE = 2.434  # data/processed/stage3_ensemble.pkl の実データdist_A平均


def train_model(train_loader, lam: float, target_std: float, seed: int, y_lookup: dict | None = None) -> TrajectoryBackbone:
    """y_lookupがNoneなら実データのtarget_defend_posを使う（通常通り）。
    与えられれば、そのkey->(n_defend,T,2)配列を教師信号として使う（合成データ実験用）。"""
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
            mask = batch["defend_mask"]
            if y_lookup is None:
                target = batch["target_defend_pos"].permute(0, 2, 1, 3)
            else:
                target = torch.stack(
                    [torch.from_numpy(y_lookup[(m, e)]).float() for m, e in zip(batch["match_id"], batch["event_id"])]
                )
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


def train_reference_theory_model(all_trajs) -> tuple[dict[tuple[str, str], np.ndarray], float]:
    """理論(modelB, lambda=0.5)を全データで1回学習し、Y_theoryを得る。"""
    ds_all = CounterAttackDataset(all_trajs)
    loader_all = DataLoader(ds_all, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
    eval_loader_all = DataLoader(ds_all, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_samples)
    target_std_all = compute_target_compactness(all_trajs, INPUT_FRAMES, side="defend")

    print(f"  training reference modelB(theory) on all data (target_std={target_std_all:.2f})...")
    model_ref = train_model(loader_all, lam=LAMBDA_B, target_std=target_std_all, seed=SEED)
    y_theory = get_predictions(model_ref, eval_loader_all)
    return y_theory, target_std_all


def add_noise(y_theory: dict, sigma: float, rng: np.random.Generator) -> dict[tuple[str, str], np.ndarray]:
    return {key: val + rng.normal(0.0, sigma, size=val.shape).astype(np.float32) for key, val in y_theory.items()}


def run_fold(train_trajs, holdout_trajs, y_synth: dict, target_std: float) -> list[dict]:
    train_ds = CounterAttackDataset(train_trajs)
    holdout_ds = CounterAttackDataset(holdout_trajs)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
    holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_samples)

    model_a = train_model(train_loader, lam=0.0, target_std=target_std, seed=SEED, y_lookup=y_synth)
    model_b = train_model(train_loader, lam=LAMBDA_B, target_std=target_std, seed=SEED, y_lookup=y_synth)

    pred_a = get_predictions(model_a, holdout_loader)
    pred_b = get_predictions(model_b, holdout_loader)

    masks = {}
    for batch in holdout_loader:
        for i in range(len(batch["match_id"])):
            key = (batch["match_id"][i], batch["event_id"][i])
            masks[key] = batch["defend_mask"][i].numpy()

    records = []
    for key in pred_a:
        m = masks[key]
        real_synth = y_synth[key]  # 「合成された真の観測」
        dist_a = np.linalg.norm(pred_a[key] - real_synth, axis=-1)[m].mean()
        dist_b = np.linalg.norm(pred_b[key] - real_synth, axis=-1)[m].mean()
        records.append({"match_id": key[0], "event_id": key[1], "dist_A": float(dist_a), "dist_B": float(dist_b),
                         "excess_deviation": float(dist_b - dist_a)})
    return records


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        all_trajs = pickle.load(f)
    rng = np.random.default_rng(0)

    print("=== 理論モデル(modelB, 全データ)を1回学習してY_theoryを生成 ===")
    y_theory, target_std = train_reference_theory_model(all_trajs)

    print("\n=== sigmaキャリブレーション（J03WPY 1foldのみ） ===")
    holdout_match = "J03WPY"
    train_trajs = [t for t in all_trajs if t.match_id != holdout_match]
    holdout_trajs = [t for t in all_trajs if t.match_id == holdout_match]

    chosen_sigma = None
    best_gap = float("inf")
    for sigma in [0.5, 1.0, 1.5, 2.0]:
        y_synth = add_noise(y_theory, sigma, rng)
        recs = run_fold(train_trajs, holdout_trajs, y_synth, target_std)
        dist_a_mean = np.mean([r["dist_A"] for r in recs])
        gap = abs(dist_a_mean - REAL_DIST_A_REFERENCE)
        print(f"  sigma={sigma:.1f}: dist_A_synth={dist_a_mean:.3f} (real dist_A={REAL_DIST_A_REFERENCE:.3f}, gap={gap:.3f})")
        if gap < best_gap:
            best_gap = gap
            chosen_sigma = sigma

    print(f"\n選択されたsigma = {chosen_sigma}\n")

    print("=== 本番：7-fold LOMO CVで超過逸脱度の構造的バイアスを測定 ===")
    y_synth = add_noise(y_theory, chosen_sigma, rng)

    all_records = []
    for holdout_match in ALL_MATCHES:
        train_trajs = [t for t in all_trajs if t.match_id != holdout_match]
        holdout_trajs = [t for t in all_trajs if t.match_id == holdout_match]
        recs = run_fold(train_trajs, holdout_trajs, y_synth, target_std)
        all_records.extend(recs)
        print(f"  holdout={holdout_match}: n={len(recs)}")

    excess = np.array([r["excess_deviation"] for r in all_records])
    dist_a = np.array([r["dist_A"] for r in all_records])
    print(f"\nn={len(all_records)}")
    print(f"dist_A_synth: mean={dist_a.mean():.3f} (参考: 実データdist_A mean={REAL_DIST_A_REFERENCE:.3f})")
    print(f"excess_deviation_synth（構造的な下駄）: mean={excess.mean():.3f} std={excess.std():.3f}")

    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / f"synthetic_bias_calibration_sigma{chosen_sigma}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"records": all_records, "sigma": chosen_sigma}, f)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
