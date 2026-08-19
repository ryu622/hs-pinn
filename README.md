# hs-pinn

サッカーのトラッキングデータ（idsse-data、Bundesliga 7試合）を用い、カウンター攻撃における**守備側の戦術セオリー逸脱**を定量化する修士研究プロジェクト。

「hs」は hard/soft constraints（ハード制約・ソフト制約）の略。選手の軌道予測モデルに、物理的に破れないハード制約（加速度・速度上限、ピッチ境界）と、セオリーとして破りうるソフト制約（損失関数へのペナルティ）を組み込み、両者の性質の違いを研究の軸に据えている。

## 研究の全体像

最新の設計・進捗・次のアクションは **[`documents/resarch_plan.md`](documents/resarch_plan.md)** に集約されている（このプロジェクトで最初に読むべきドキュメント）。現時点までの手法・主結果・課題・今後の方針の総括は **[`documents/thesis_theme_summary_report.md`](documents/thesis_theme_summary_report.md)** にまとまっている。

研究の骨子（詳細はresarch_plan.md 10節）：

1. **modelA**（データ駆動ghost、λ=0）：実際の守備行動を模倣する、理論なしのモデル
2. **modelB**（理論駆動ghost、λ>0）：同一アーキテクチャに、検証したいセオリー（守備側コンパクトネス）を損失関数として加えたモデル
3. **超過逸脱度** = d(実観測, modelB) − d(実観測, modelA)：一般的な予測誤差を差し引き、理論からの逸脱そのものを分離して測定する

この指標を、3段階ゲート（観測データへの直接相関 → 損失関数としての機能スクリーニング → modelA/B構築とLeave-One-Match-Out交差検証）で検証し、カウンター攻撃の成功/失敗との関係を調べている。

## 環境構築

`uv` でPython環境・依存関係を管理する（詳細は[`CLAUDE.md`](CLAUDE.md)）。

```bash
uv pip install -r requirements.txt   # 依存関係のインストール（初回）
uv run python <script>.py            # スクリプトの実行（必ずuv run経由）
uv run pytest                        # テストの実行
```

## ディレクトリ構成

```
src/hs_pinn/        # コアライブラリ（データ読み込み・モデル・制約・指標）
  counter_events.py     カウンター攻撃候補イベントの抽出（kloppy経由）
  trajectories.py       CounterTrajectory（全パイプライン共通の中間データ形式）
  dataset.py             学習用テンソルへの変換
  model.py                TrajectoryBackbone（Self-Attention + GRU + 制約層）
  hard_constraints.py     ハード制約層（加速度/速度上限、ピッチ境界）
  soft_constraints.py     ソフト制約（コンパクトネス損失など）
  tactic_metrics.py       コンパクトネス等の戦術指標
  space_control.py        スペース支配（Voronoi等）の指標

scripts/             各検証ステップの実行スクリプト（段階1〜3、λスイープ、
                       アンサンブル、クラスタロバスト検定、合成データによる
                       バイアスキャリブレーション等）

documents/            進捗・検証結果のレポート群（append-only、resarch_plan.md
                       が索引）

tests/                pytest（hard_constraints, model, soft_constraintsの
                       単体テスト）

data/                 生成物（gitignore対象、scripts/build_dataset.py 等で再生成）
```

## データ読み込み層を差し替える場合

`CounterTrajectory`（`src/hs_pinn/trajectories.py`）を共通の中間形式として、それ以降（モデル・学習・検証パイプライン全体）が分離された設計になっている。新しいトラッキングデータ形式（kloppy非対応のものを含む）を使う場合も、そのデータから`CounterTrajectory`を構築する変換層を実装すれば、以降のパイプラインはそのまま再利用できる。
