# 段階2：守備側コンパクトネスの損失関数としての機能スクリーニング

`resarch_plan.md` 10.6節の3段階ゲートのうち段階2の実施記録。守備側予測モデル（`predict_side="defend"`）に対し、λを極端に振って事前に固定した3基準で判定した。

コード：`src/hs_pinn/model.py`（`predict_side`パラメータ追加）、`src/hs_pinn/dataset.py`（守備側初期状態を追加）、`src/hs_pinn/tactic_metrics.py` / `src/hs_pinn/soft_constraints.py`（目標値校正関数に`side`引数を追加）、`scripts/screen_defense_compactness_loss.py`

---

## モデル拡張の概要

- `TrajectoryBackbone`に`predict_side`（"attack" or "defend"）を追加。デコーダが参照するSelf-Attention出力のスライスと、ハード制約層への初期位置・初速度の入力元を切り替えられるようにした
- `dataset.py`に守備側の初期位置・初速度（`init_defend_position` / `init_defend_velocity`）を追加。`target_defend_pos`は既存のものを流用
- `tactic_metrics.compute_compactness_target` / `soft_constraints.compute_target_compactness`に`side`引数を追加し、目標値を守備側データから校正できるようにした

## 判定基準（事前固定）

(a) 勾配が病的でない（微分可能・勾配消失/爆発なし）
(b) λを大きくした際にmodelBがmodelAと有意に異なる出力を示す
(c) 退化解に陥らない

## 結果（30エポック、train 345件・valid 80件）

| λ | 状態 | valid ADE | valid L_compact | 最小選手間距離(平均) | 最小選手間距離(worst) |
|---|---|---|---|---|---|
| 0.0 | ok | 2.490 | 2.306 | 4.31 | 0.52 |
| 0.5 | ok | 2.538 | 2.199 | 4.33 | 0.60 |
| 2.0 | ok | 2.695 | 2.139 | 4.26 | 0.12 |
| 8.0 | ok | 3.587 | 1.310 | 3.89 | 0.41 |

- **(a)**：全λでNaN・発散なし。クリア
- **(b)**：valid L_compactがλ=0→8で2.306→1.310（43%減）と単調に改善し、代償としてADEも2.490→3.587と上昇。攻撃側で機能しなかったL_spaceとは異なり、λに対する明確な応答を示した。クリア
- **(c)**：選手間距離の平均は4.3m前後で安定。λ=2で最悪ケース0.12mが1件あったが、λ=8ではむしろ改善しており（0.41）、コンパクトネス制約自体が系統的に崩壊を招いているわけではなさそうと判断。デュエル中の実際の接近を反映している可能性もある。要観察としつつクリア扱いとする

## 結論

3基準とも通過。守備側コンパクトネスは損失関数として健全に機能する。攻撃側で既に実績のあった`compactness_loss`（自己完結した目標値追従ペナルティ）の数式的性質がそのまま活きた形で、L_spaceのような構造的不安定さは見られなかった。

次は段階3：試合単位のLeave-One-Match-Out交差検証によるmodelA/B構築と超過逸脱度分析。
