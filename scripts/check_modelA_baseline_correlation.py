"""課題A対処法2（計画書10.11節・10.10節項目13）：modelA単独の予測誤差と成功率の対照相関。

超過逸脱度＝dist_B - dist_Aが成功率と相関するとして、それが「理論への違反」
ではなく「そもそも予測しにくい（非定型な）守備だったから」という一般的な
予測困難性を捉えているだけの可能性がある。この懸念を確認するため、理論なし
の模倣モデル単体（modelA、λ=0）の予測誤差(dist_A)だけで、成功率とどの程度
相関するかを見る。dist_Aがそれ単体で強く成功率と相関していれば、超過逸脱度の
解釈には注意が必要（一般的な非定型性の寄与を疑うべき）。

実行: uv run python scripts/check_modelA_baseline_correlation.py [path/to/records.pkl]
（省略時はdata/processed/stage3_ensemble.pkl、λ=0.5・アンサンブル版の主結果）
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats

DEFAULT_RECORDS = Path(__file__).resolve().parent.parent / "data" / "processed" / "stage3_ensemble.pkl"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RECORDS
    with open(path, "rb") as f:
        records = pickle.load(f)

    labels = np.array([r["label"] for r in records])
    dist_a = np.array([r["dist_A"] for r in records])
    excess = np.array([r["excess_deviation"] for r in records])
    recovery_x = np.array([r["recovery_x"] for r in records])

    print(f"=== {path.name} (n={len(records)}) ===\n")
    print(f"dist_A: mean={dist_a.mean():.3f} std={dist_a.std():.3f}")
    print(f"excess_deviation: mean={excess.mean():.3f} std={excess.std():.3f}")
    print(f"corr(dist_A, excess_deviation) = {np.corrcoef(dist_a, excess)[0, 1]:+.3f}\n")

    # 生の相関：modelA単独の誤差 vs 成功率
    r_raw, p_raw = stats.pointbiserialr(labels, dist_a)
    print(f"[modelA単独] 生の相関(交絡無視): r={r_raw:+.3f}  p={p_raw:.4f}")

    # 交絡制御（奪取位置）した上での関係
    X = np.column_stack([dist_a, recovery_x])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = sm.add_constant(X)
    model = sm.Logit(labels, X).fit(disp=0)
    print(f"[modelA単独] ロジスティック回帰(奪取位置で制御): beta={model.params[1]:+.3f}  p={model.pvalues[1]:.4f}")

    print()
    print("--- 比較: 超過逸脱度(理論反映済み)の同じ検定 ---")
    r_raw_e, p_raw_e = stats.pointbiserialr(labels, excess)
    print(f"[超過逸脱度] 生の相関(交絡無視): r={r_raw_e:+.3f}  p={p_raw_e:.4f}")
    X2 = np.column_stack([excess, recovery_x])
    X2 = (X2 - X2.mean(axis=0)) / X2.std(axis=0)
    X2 = sm.add_constant(X2)
    model2 = sm.Logit(labels, X2).fit(disp=0)
    print(f"[超過逸脱度] ロジスティック回帰(奪取位置で制御): beta={model2.params[1]:+.3f}  p={model2.pvalues[1]:.4f}")


if __name__ == "__main__":
    main()
