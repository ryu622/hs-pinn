"""超過逸脱度の交絡制御ロジスティック回帰（段階3の分析部分を再利用可能にしたもの）。

stage3_ensemble.py / stage3_excess_deviation.py が出力するrecordsのpickle
（match_id, event_id, label, recovery_x, excess_deviationを含む）を受け取り、
奪取位置(recovery_x)を制御した上で超過逸脱度が成功率と関係するかを確認する。

実行: uv run python scripts/analyze_excess_deviation.py <path/to/records.pkl>
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats


def analyze(records: list[dict]) -> None:
    labels = np.array([r["label"] for r in records])
    excess = np.array([r["excess_deviation"] for r in records])
    recovery_x = np.array([r["recovery_x"] for r in records])

    print(f"n = {len(records)}, success rate = {labels.mean():.3f}")
    print(f"excess_deviation: mean={excess.mean():.3f} std={excess.std():.3f}")

    r_raw, p_raw = stats.pointbiserialr(labels, excess)
    print(f"\n生の相関(交絡無視): r={r_raw:+.3f}  p={p_raw:.4f}")

    X = np.column_stack([excess, recovery_x])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = sm.add_constant(X)
    model = sm.Logit(labels, X).fit(disp=0)
    beta_excess, p_excess = model.params[1], model.pvalues[1]
    beta_recovery, p_recovery = model.params[2], model.pvalues[2]

    print("ロジスティック回帰(標準化係数、奪取位置で制御):")
    print(f"  超過逸脱度:     beta={beta_excess:+.3f}  p={p_excess:.4f}")
    print(f"  奪取位置(参考):  beta={beta_recovery:+.3f}  p={p_recovery:.4f}")


def main() -> None:
    path = Path(sys.argv[1])
    with open(path, "rb") as f:
        records = pickle.load(f)
    print(f"=== {path.name} ===")
    analyze(records)


if __name__ == "__main__":
    main()
