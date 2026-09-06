"""「なぜPINNが必要か」の3者比較ケーススタディ（計画書11.4節・11.9節項目2）。

①PINNゴースト（modelB、ハード制約+ソフト制約）②ナイーブな理論最適化ゴースト
（学習なし、コンパクトネスを純粋に数理最適化しただけの軌道）③modelA（データ駆動、
理論制約なし）の3者を代表シーンで並べて可視化し、「PINNゴーストは③ほど自由では
ないが、②のような不自然さもなく、セオリー通りかつ現実的にありえる動きになって
いる」ことを定性的に示す。

この比較の目的は理論を強く効かせたときのPINNの優位性を示すストレステストであり、
ランキングに使う運用上のλ（0.5）とは役割が異なるため、意図的に強めのλ=8を使う
（11.5節）。

【修正履歴】初期版は、①事例選定に使うdist_Bランキング（λ=0.5・LOMO-CVアンサンブル、
excess_deviation_aggregated.pkl由来）と、②実際にmodelBパネルに描画するモデル
（このスクリプト内で全502件をin-sample学習したλ=8・単一seed）が別物になっており、
「ランキング上位として選んだのに、可視化している乖離とランキングの根拠が一致しない」
という不整合があった（`coaching_overlay_prototype_report.md`で発見）。本版では、
①②とも同一のλ=8・LOMO-CV（該当試合をホールドアウト）モデルで統一する。

事例選定は、この統一したλ=8・LOMO-CVのdist_Bランキングの上位から、試合の多様性を
確保して3件選ぶ。

実行: uv run python scripts/three_way_comparison.py
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
    TARGET_FRAMES,
    CounterAttackDataset,
    collate_samples,
)
from hs_pinn.hard_constraints import HardConstraintLayer, PitchBounds
from hs_pinn.model import TrajectoryBackbone
from hs_pinn.soft_constraints import compactness_loss, compute_target_compactness

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
OUT_DIR = Path(__file__).resolve().parent.parent / "documents" / "three_way_comparison_images"
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
PITCH_BOUNDS = PitchBounds(0.0, PITCH_LENGTH, 0.0, PITCH_WIDTH)
LAMBDA_STRONG = 8.0  # ストレステスト用（運用上のlambda=0.5とは別に、意図的に強く効かせる）
EPOCHS = 30
BATCH_SIZE = 16
NAIVE_STEPS = 300
N_EVENTS = 3
ALL_MATCHES = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]


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


def naive_optimization_ghost(init_pos: torch.Tensor, mask: torch.Tensor, target_std: float) -> np.ndarray:
    """学習なし、コンパクトネス損失だけを直接最適化するナイーブなゴースト。

    データ項（実軌道への忠実さ）もハード制約（速度・加速度上限）も一切課さない。
    init_pos: (n_defend, 2)。最終観測フレームの実位置を初期値として複製する。
    戻り値: (n_defend, T_target, 2)
    """
    n_defend = init_pos.shape[0]
    pos = init_pos.unsqueeze(0).repeat(TARGET_FRAMES, 1, 1).clone().unsqueeze(0)  # (1, T, n, 2)
    pos = pos.permute(0, 2, 1, 3).contiguous()  # (1, n, T, 2)
    pos.requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=0.3)
    mask_b = mask.unsqueeze(0)
    for _ in range(NAIVE_STEPS):
        loss = compactness_loss(pos, mask_b, target_std)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return pos[0].detach().numpy()  # (n, T, 2)


def plot_comparison(tag: str, match_id: str, event_id: str, real_defend, pred_a, pred_b_strong, pred_naive,
                     attack_real, defend_mask, ball_pos, dist_b: float) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(30, 6.8))
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, line_color="black")

    panels = [
        ("REAL defense", real_defend),
        ("modelA (data-driven, no theory)", pred_a),
        (f"modelB PINN ghost (lambda={LAMBDA_STRONG:.0f})", pred_b_strong),
        ("naive optimization ghost (no learning)", pred_naive),
    ]

    for ax, (title, defend_traj) in zip(axes, panels):
        pitch.draw(ax=ax)
        for i in range(attack_real.shape[1]):
            xs, ys = attack_real[:, i, 0], attack_real[:, i, 1]
            if (xs == 0).all() and (ys == 0).all():
                continue
            ax.plot(xs, ys, color="crimson", alpha=0.3, linewidth=1.0, zorder=1)
            ax.scatter(xs[-1], ys[-1], color="crimson", s=30, zorder=2, marker="^", alpha=0.6)

        for i in range(defend_traj.shape[0]):
            if not defend_mask[i]:
                continue
            xs, ys = defend_traj[i, :, 0], defend_traj[i, :, 1]
            ax.plot(xs, ys, color="royalblue", alpha=0.7, linewidth=1.5, zorder=3)
            ax.scatter(xs[0], ys[0], color="royalblue", s=50, zorder=4, marker="o")
            ax.scatter(xs[-1], ys[-1], color="royalblue", s=80, zorder=4, marker="^")

        ax.plot(ball_pos[:, 0], ball_pos[:, 1], color="black", linewidth=1.5, linestyle="--", zorder=5)
        ax.scatter(ball_pos[0, 0], ball_pos[0, 1], color="black", s=40, zorder=6, marker="o", facecolors="none")
        ax.scatter(ball_pos[-1, 0], ball_pos[-1, 1], color="black", s=40, zorder=6, marker="^")
        ax.set_title(title, fontsize=11)

    fig.suptitle(
        f"[{tag}] {match_id} / {event_id}  dist_B(lambda={LAMBDA_STRONG:.0f}, LOMO-CV, same model as ranking)={dist_b:.2f}\n"
        f"(red=attack(real, context), blue=defense)",
        fontsize=13,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{tag}_{match_id}_{event_id}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        all_trajs = pickle.load(f)
    traj_by_key = {(t.match_id, t.event_id): t for t in all_trajs}

    # ランキングと可視化を同一モデル(λ=8・LOMO-CV)で統一するため、全7試合分を
    # ホールドアウト方式で学習・予測し、real/modelA/modelB(λ=8)の軌道と
    # dist_Bをすべて先にキャッシュしてから事例選定する。
    cache: dict[tuple[str, str], dict] = {}
    labels_by_key: dict[tuple[str, str], int] = {}

    for holdout_match in ALL_MATCHES:
        train_trajs = [t for t in all_trajs if t.match_id != holdout_match]
        holdout_trajs = [t for t in all_trajs if t.match_id == holdout_match]

        train_ds = CounterAttackDataset(train_trajs)
        holdout_ds = CounterAttackDataset(holdout_trajs)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_samples)
        holdout_loader = DataLoader(holdout_ds, batch_size=1, shuffle=False, collate_fn=collate_samples)

        target_std = compute_target_compactness(train_trajs, INPUT_FRAMES, side="defend")
        print(f"[holdout={holdout_match}] target_std={target_std:.2f}  training modelA(lambda=0)...")
        model_a = train_model(train_loader, lam=0.0, target_std=target_std)
        print(f"[holdout={holdout_match}] training modelB(lambda={LAMBDA_STRONG})...")
        model_b = train_model(train_loader, lam=LAMBDA_STRONG, target_std=target_std)

        for batch in holdout_loader:
            key = (batch["match_id"][0], batch["event_id"][0])
            with torch.no_grad():
                pos_a, _, _ = model_a(batch)
                pos_b, _, _ = model_b(batch)
            defend_mask = batch["defend_mask"][0].numpy()
            real_defend = batch["target_defend_pos"].permute(0, 2, 1, 3)[0].numpy()  # (n_defend, T, 2)
            pred_a = pos_a[0].numpy()
            pred_b = pos_b[0].numpy()
            diff_b = np.linalg.norm(pred_b - real_defend, axis=-1)
            dist_b = float(diff_b[defend_mask].mean())

            cache[key] = {
                "real_defend": real_defend,
                "pred_a": pred_a,
                "pred_b": pred_b,
                "defend_mask": defend_mask,
                "attack_real": batch["target_attack_pos"][0].numpy(),
                "init_defend_position": batch["init_defend_position"][0],
                "dist_b": dist_b,
                "target_std": target_std,
            }
            labels_by_key[key] = int(batch["label"][0])

    # ボールがほぼ動いていない（デッドボール）事例を除外する。事例3(J03WR9/18242100001234)で、
    # ボールが4秒間ピッチ外(-1.31, 39.73)で完全に静止しているデッドボール局面が、
    # dist_Bランキング上位に紛れ込んでいたことが判明したため(ユーザー指摘で発見)。
    MIN_BALL_PATH_LENGTH = 5.0  # メートル、予測ホライズン4秒間の総移動距離

    def ball_path_length(key: tuple[str, str]) -> float:
        ball = traj_by_key[key].ball_pos[INPUT_FRAMES:]
        diffs = np.linalg.norm(np.diff(ball, axis=0), axis=1)
        return float(np.nansum(diffs))

    records = [
        {"match_id": k[0], "event_id": k[1], "dist_B": v["dist_b"], "label": labels_by_key[k],
         "ball_path_length": ball_path_length(k)}
        for k, v in cache.items()
    ]
    n_before = len(records)
    records = [r for r in records if r["ball_path_length"] >= MIN_BALL_PATH_LENGTH]
    print(f"ball-activity filter: {n_before} -> {len(records)} events (excluded {n_before - len(records)} near-dead-ball events)")

    # dist_Bランキング上位から、試合の多様性を確保してN_EVENTS件選ぶ
    records_sorted = sorted(records, key=lambda r: -r["dist_B"])
    selected, seen_matches = [], set()
    for r in records_sorted:
        if r["match_id"] in seen_matches:
            continue
        selected.append(r)
        seen_matches.add(r["match_id"])
        if len(selected) >= N_EVENTS:
            break

    # 追加: 逸脱度が最も高い「成功」シーン（既存の上位はすべて失敗だったため、対照として追加）
    success_sorted = [r for r in records_sorted if r["label"] == 1]
    top_success = success_sorted[0]
    if (top_success["match_id"], top_success["event_id"]) not in {(r["match_id"], r["event_id"]) for r in selected}:
        selected.append(top_success)

    for r in selected:
        print(f"selected: {r['match_id']} / {r['event_id']}  dist_B={r['dist_B']:.3f}  label={r['label']}")

    for r in selected:
        key = (r["match_id"], r["event_id"])
        c = cache[key]
        pred_naive = naive_optimization_ghost(c["init_defend_position"], torch.from_numpy(c["defend_mask"]), c["target_std"])
        ball_pos = traj_by_key[key].ball_pos[INPUT_FRAMES:]

        path = plot_comparison(
            "three_way", r["match_id"], r["event_id"], c["real_defend"], c["pred_a"], c["pred_b"], pred_naive,
            c["attack_real"], c["defend_mask"], ball_pos, r["dist_B"],
        )
        print(f"saved {path}")


if __name__ == "__main__":
    main()
