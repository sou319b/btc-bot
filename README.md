# 暗号資産取引システム

## システム概要
このシステムは、機械学習を活用した暗号資産（仮想通貨）の自動取引システムです。テクニカル分析と機械学習モデルを組み合わせて、取引判断を行います。
ためしでつくったもの全然うごかん

## ファイル構成と目的

### コアファイル
1. `p19_strategy.py`
   - 基本的な取引戦略のコアロジックを実装
   - RSI、ボリンジャーバンド、移動平均線などの技術的分析を組み合わせた取引判断
   - ポジションサイズの最適化、損切り・利確の判断ロジックを含む

2. `p19_ml_strategy.py`
   - 機械学習モデルを統合した取引戦略の実装
   - 従来の技術的分析と機械学習モデルの予測を組み合わせた取引判断
   - モデルの予測を重み付けして総合的な判断を行う

3. `ml_strategy_patterns.py`
   - 機械学習戦略のパターン定義
   - 予測目標、特徴量、モデルタイプ、時間枠などの設定
   - 様々な戦略パターンの組み合わせを提供

### データ処理・学習関連ファイル
4. `p19_fetch_data_v2.py`
   - Bybit APIを使用して価格データ（OHLCV）を取得
   - 1分足の過去1週間分のデータを取得
   - データの検証と前処理を実施

5. `p19_feature_engineering.py`
   - 価格データから特徴量を生成
   - テクニカル指標の計算
   - 市場特性の分析と特徴量化

6. `p19_model_training.py`
   - 機械学習モデルの学習を実行
   - 複数の戦略パターンに基づくモデルを学習
   - モデルの保存と評価結果の出力

7. `p19_model_evaluation.py`
   - 学習したモデルの性能評価
   - 分類・回帰モデルの評価指標の計算
   - 評価結果の可視化と保存

## 実行順序

1. データ取得
```bash
python p19_fetch_data_v2.py
```
- `./data/historical_data_[日付].csv` にデータが保存されます

2. 特徴量エンジニアリング
```bash
python p19_feature_engineering.py
```
- `./data/features_[日付].csv` に特徴量が保存されます

3. モデル学習
```bash
python p19_model_training.py
```
- `./models/` ディレクトリに学習済みモデルが保存されます
- `./results/training_results.json` に学習結果が保存されます

4. モデル評価
```bash
python p19_model_evaluation.py
```
- `./results/evaluation_report.json` に評価結果が保存されます
- `./results/evaluation_plots/` に評価プロットが保存されます

## 必要な環境設定

### 必要なパッケージ
```
pandas
numpy
scikit-learn
tensorflow
xgboost
pandas-ta
pybit
joblib
```

インストール方法：
```bash
pip install -r requirements.txt
```

### ディレクトリ構造
```
.
├── data/                  # データ保存ディレクトリ
├── models/               # 学習済みモデル保存ディレクトリ
├── results/              # 結果保存ディレクトリ
└── logs/                 # ログファイル保存ディレクトリ
```

### 注意事項
- Bybit APIを使用するため、インターネット接続が必要です
- GPUが利用可能な場合、TensorFlowは自動的にGPUを使用します
- メモリ使用量に注意してください（特に学習時）
- すべてのディレクトリは自動的に作成されます

## エラー対処
一般的なエラーとその解決方法：

1. データ取得エラー
   - インターネット接続を確認
   - API制限に注意

2. メモリエラー
   - バッチサイズの調整
   - 不要なプロセスの終了

3. GPU関連エラー
   - TensorFlowのバージョン確認
   - GPUドライバーの更新

## 拡張と改善
システムの改善のためのヒント：

1. 新しい特徴量の追加
   - `p19_feature_engineering.py` に新しい特徴量を追加
   - `ml_strategy_patterns.py` の特徴量パターンを更新

2. 新しい戦略パターンの追加
   - `ml_strategy_patterns.py` に新しいパターンを追加
   - 既存のパターンのパラメータを調整

3. モデルのハイパーパラメータ調整
   - `p19_model_training.py` でモデルのパラメータを調整
   - クロスバリデーションの追加

