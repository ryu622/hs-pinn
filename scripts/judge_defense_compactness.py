"""段階1：守備側コンパクトネスの生の相関チェック（計画書10.6節）。

モデル学習なし、観測データに直接計算する。奪取位置という交絡変数を
ロジスティック回帰で制御した上で、守備側コンパクトネスが独立に
成功率と関係するかを確認する。

実行: uv run python scripts/judge_defense_compactness.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats

from hs_pinn.tactic_metrics import COMPACTNESS_FORMULATIONS, compactness_scalar

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)

    labels = np.array([t.label for t in trajs])
    # 奪取位置：観測窓の最初のフレーム(t=0、奪取の瞬間)のボールx座標。
    # 攻撃方向は常に+xに正規化済みなので、値が大きいほど奪取地点が相手ゴールに近い。
    recovery_x = np.array([t.ball_pos[0, 0] for t in trajs])

    print(f"n = {len(trajs)}, success rate = {labels.mean():.3f}")
    print(f"recovery_x: mean={recovery_x.mean():.1f} std={recovery_x.std():.1f}\n")

    scores: dict[str, np.ndarray] = {}

    print("=" * 78)
    print("守備側コンパクトネス：判定①（分布）・判定②（生の相関 vs 交絡制御後）")
    print("=" * 78)
    for name in COMPACTNESS_FORMULATIONS:
        vals = np.array([compactness_scalar(t.defend_pos, name) for t in trajs])
        scores[name] = vals
        valid = ~np.isnan(vals)

        # 判定②：生の相関（交絡変数を無視）
        r_raw, p_raw = stats.pointbiserialr(labels[valid], vals[valid])

        # 交絡制御：奪取位置を含めたロジスティック回帰
        X = np.column_stack([vals[valid], recovery_x[valid]])
        X = (X - X.mean(axis=0)) / X.std(axis=0)  # 係数を比較しやすいよう標準化
        X = sm.add_constant(X)
        y = labels[valid]
        model = sm.Logit(y, X).fit(disp=0)
        beta_compact, p_compact = model.params[1], model.pvalues[1]
        beta_recovery, p_recovery = model.params[2], model.pvalues[2]

        print(f"\n{name}")
        print(f"  n_nan={(~valid).sum()}  mean={np.nanmean(vals):.2f}  std={np.nanstd(vals):.2f}")
        print(f"  生の相関(交絡無視):        r={r_raw:+.3f}  p={p_raw:.4f}")
        print(f"  ロジスティック回帰(標準化係数、奪取位置で制御):")
        print(f"    コンパクトネス: beta={beta_compact:+.3f}  p={p_compact:.4f}")
        print(f"    奪取位置(参考):  beta={beta_recovery:+.3f}  p={p_recovery:.4f}")

    print()
    print("=" * 78)
    print("判定③：守備側3定式化間の相関")
    print("=" * 78)
    names = list(scores.keys())
    header = " " * 30 + " ".join(f"{n[:14]:>15s}" for n in names)
    print(header)
    for ni in names:
        row = [f"{ni:30s}"]
        for nj in names:
            a, b = scores[ni], scores[nj]
            valid = ~(np.isnan(a) | np.isnan(b))
            r, _ = stats.pearsonr(a[valid], b[valid])
            row.append(f"{r:+15.2f}")
        print(" ".join(row))

    out_path = CACHE_PATH.parent / "defense_compactness_scores.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"labels": labels, "recovery_x": recovery_x, "scores": scores}, f)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
