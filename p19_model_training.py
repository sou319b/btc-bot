"""
機械学習モデルの学習スクリプト
各種モデルのトレーニングと保存を行う
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'p19_ml'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow警告の抑制

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import tensorflow as tf
from ml_strategy_patterns import (
    StrategyPattern, PredictionTargets, FeaturePatterns,
    MLModels, TimeFrames, STRATEGY_PATTERNS
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import traceback

class ModelTrainer:
    def __init__(self, df: pd.DataFrame, strategy: StrategyPattern):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            特徴量を含むデータフレーム
        strategy : StrategyPattern
            学習する戦略パターン
        """
        self.df = df.copy()
        self.strategy = strategy
        self.model = None
        self.scaler = None
        self.feature_scaler = None
    
    def initialize_model(self) -> object:
        """モデルの初期化"""
        # モデルタイプの正規化
        model_type = self.strategy.model_type.lower()
        
        if 'random_forest' in model_type:
            if self.is_classification_task():
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            else:
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
        
        elif 'xgboost' in model_type:
            if self.is_classification_task():
                return XGBClassifier(
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    objective='multi:softmax' if self.is_classification_task() else 'reg:squarederror'
                )
            else:
                return XGBRegressor(
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
        
        elif 'lstm' in model_type:
            sequence_length = 10
            n_features = len(self.strategy.features)
            
            from tensorflow.keras.layers import Input
            inputs = Input(shape=(sequence_length, n_features))
            x = LSTM(50, return_sequences=True)(inputs)
            x = Dropout(0.2)(x)
            x = LSTM(50, return_sequences=False)(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1, activation='sigmoid' if self.is_classification_task() else None)(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy' if self.is_classification_task() else 'mse',
                metrics=['accuracy'] if self.is_classification_task() else None
            )
            return model
        
        raise ValueError(f"未対応のモデルタイプです: {self.strategy.model_type}")
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """データの準備"""
        try:
            # 特徴量の存在確認
            missing_features = [f for f in self.strategy.features if f not in self.df.columns]
            if missing_features:
                # One-Hotエンコーディング後の特徴量名に対応
                new_features = []
                for f in self.strategy.features:
                    if f in missing_features:
                        # カテゴリカル変数の場合、One-Hotエンコーディング後のカラムを探す
                        matching_cols = [col for col in self.df.columns if col.startswith(f + '_')]
                        if matching_cols:
                            new_features.extend(matching_cols)
                        else:
                            raise ValueError(f"特徴量 {f} に対応するOne-Hotエンコーディングされたカラムが見つかりません")
                    else:
                        new_features.append(f)
                self.strategy.features = new_features
            
            # 特徴量とターゲットの分離
            X = self.df[self.strategy.features].values
            
            # ターゲット変数の処理
            if self.strategy.prediction_target not in self.df.columns:
                # One-Hotエンコーディング後のターゲット変数を探す
                target_cols = [col for col in self.df.columns if col.startswith(self.strategy.prediction_target + '_')]
                if target_cols:
                    # One-Hotエンコーディングされた列から元のカテゴリを復元
                    y = np.argmax(self.df[target_cols].values, axis=1)
                else:
                    raise ValueError(f"予測対象 {self.strategy.prediction_target} が見つかりません")
            else:
                y = self.df[self.strategy.prediction_target].values
            
            # 分類タスクの場合の処理
            if self.is_classification_task():
                if self.strategy.prediction_target == PredictionTargets.TREND_STRENGTH:
                    # トレンド強度を3つのカテゴリに分類
                    y = pd.qcut(y, q=3, labels=[0, 1, 2]).astype(int)
                else:
                    # カテゴリ値を連続的な整数に変換
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
            
            # スケーリング
            if self.strategy.preprocessing['scaler'] == 'standard':
                self.feature_scaler = StandardScaler()
            else:
                self.feature_scaler = MinMaxScaler()
            
            X = self.feature_scaler.fit_transform(X)
            
            # ターゲットのスケーリング（回帰問題の場合）
            if not self.is_classification_task():
                self.scaler = MinMaxScaler()
                y = self.scaler.fit_transform(y.reshape(-1, 1)).ravel()
            
            return X, y
            
        except Exception as e:
            raise ValueError(f"データの準備中にエラーが発生しました: {str(e)}")
    
    def prepare_lstm_data(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """LSTMのための時系列データ準備"""
        X_seq = []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:(i + sequence_length)])
        return np.array(X_seq)
    
    def train_model(self) -> Tuple[object, Dict]:
        """モデルの学習"""
        # モデルの初期化
        self.model = self.initialize_model()
        if self.model is None:
            raise ValueError(f"モデルの初期化に失敗しました: {self.strategy.model_type}")
        
        # データの準備
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # LSTMの場合、データの形状を変更
        if self.strategy.model_type == 'lstm':
            sequence_length = 10
            X_train = self.prepare_lstm_data(X_train, sequence_length)
            X_test = self.prepare_lstm_data(X_test, sequence_length)
            y_train = y_train[sequence_length:]
            y_test = y_test[sequence_length:]
        
        # モデルの学習
        if self.strategy.model_type == 'lstm':
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
        else:
            self.model.fit(X_train, y_train)
        
        # 評価
        metrics = self.evaluate_model(X_test, y_test)
        
        return self.model, metrics
    
    def is_classification_task(self) -> bool:
        """分類タスクかどうかを判定"""
        return self.strategy.prediction_target in [
            PredictionTargets.PRICE_DIRECTION,
            PredictionTargets.TREND_STRENGTH,
            PredictionTargets.TRADING_SIGNAL,
            PredictionTargets.VOLATILITY_LEVEL
        ]
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """モデルの評価"""
        if self.strategy.model_type == 'lstm':
            y_pred = self.model.predict(X_test)
        else:
            y_pred = self.model.predict(X_test)
        
        metrics = {}
        if self.is_classification_task():
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
        else:
            if self.scaler:
                y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, y_pred)
        
        return metrics
    
    def save_model(self, base_path: str = './models'):
        """モデルの保存"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        model_path = f"{base_path}/{self.strategy.name}"
        
        # モデルの保存
        if 'lstm' in self.strategy.model_type.lower():
            self.model.save(f"{model_path}_model.keras")  # .keras拡張子を使用
        else:
            joblib.dump(self.model, f"{model_path}_model.joblib")
        
        # スケーラーの保存
        if self.feature_scaler:
            joblib.dump(self.feature_scaler, f"{model_path}_feature_scaler.joblib")
        if self.scaler:
            joblib.dump(self.scaler, f"{model_path}_target_scaler.joblib")
        
        # 設定の保存
        config = {
            'name': self.strategy.name,
            'prediction_target': self.strategy.prediction_target,
            'features': self.strategy.features,
            'model_type': self.strategy.model_type,
            'time_frame': self.strategy.time_frame,
            'preprocessing': self.strategy.preprocessing
        }
        
        with open(f"{model_path}_config.json", 'w') as f:
            json.dump(config, f, indent=4)

def train_all_models(df: pd.DataFrame, strategies: List[StrategyPattern]) -> Dict[str, Dict]:
    """すべてのモデルを学習"""
    results = {}
    
    print("\n学習の進捗状況:")
    print("="*50)
    
    for strategy in strategies:
        print(f"{strategy.name}: ", end="")
        trainer = ModelTrainer(df, strategy)
        
        try:
            model, metrics = trainer.train_model()
            trainer.save_model()
            
            results[strategy.name] = {
                'metrics': metrics,
                'features': strategy.features,
                'model_type': strategy.model_type,
                'time_frame': strategy.time_frame
            }
            
            print("✓")
            
        except Exception as e:
            print(f"✗ ({str(e)})")
            continue
    
    print("="*50)
    return results

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """データの前処理"""
    # timestampをdatetimeに変換
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # カテゴリカルデータのエンコーディング
    categorical_columns = [
        'volatility_regime',
        'trend_regime',
        'market_phase',
        'trend_strength_target',
        'volatility_level'
    ]
    
    # 数値データと非数値データを分離
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 数値データの異常値処理
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    
    # 数値データの欠損値を平均値で補完
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # カテゴリカル変数の処理
    for col in categorical_columns:
        if col in df.columns:
            # カテゴリカル変数をLabelEncoderで数値に変換
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # カテゴリカルデータのOne-Hotエンコーディング
    df = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)
    
    return df

def setup_logging():
    """ロギングの設定"""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/model_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_preprocessed_data():
    """前処理済みデータの読み込み"""
    data_dir = "data"
    data_files = [f for f in os.listdir(data_dir) if f.startswith("preprocessed_data_")]
    if not data_files:
        raise FileNotFoundError("前処理済みデータファイルが見つかりません")
    
    latest_file = max(data_files)
    return pd.read_csv(os.path.join(data_dir, latest_file))

def prepare_features_and_target(df, prediction_window=1):
    """特徴量とターゲットの準備（改善版）"""
    # 特徴量の選択
    feature_columns = [
        'price_range', 'body_ratio',
        'sma_20', 'sma_50', 'sma_100',
        'ema_20', 'ema_50', 'ema_100',
        'slope_20', 'slope_50', 'slope_100',
        'volume_sma_15', 'volume_ratio',
        'momentum_15', 'momentum_30',
        'roc_15', 'roc_30',
        'atr_20'
    ]
    
    # NaNを含む行を削除
    df = df.dropna(subset=feature_columns + ['target'])
    
    # 無限大の値を含む行を削除
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_columns + ['target'])
    
    # ターゲットが-1, 0, 1の形式であることを確認し、0, 1, 2に変換
    if not df['target'].isin([-1, 0, 1]).all():
        raise ValueError("ターゲット変数が予期しない値を含んでいます")
    
    # ターゲットを0, 1, 2に変換
    target_mapping = {-1: 0, 0: 1, 1: 2}
    df['target'] = df['target'].map(target_mapping)
    
    return df[feature_columns], df['target']

def train_model():
    """モデルのトレーニングメイン関数（改善版）"""
    logger = setup_logging()
    
    try:
        # データの読み込み
        logger.info("前処理済みデータを読み込みます...")
        df = load_preprocessed_data()
        
        # 特徴量とターゲットの準備
        logger.info("特徴量とターゲットを準備します...")
        X, y = prepare_features_and_target(df)
        
        # 時系列分割の設定（改善版）
        tscv = TimeSeriesSplit(
            n_splits=3,  # 分割数を3に減らす
            test_size=int(len(X) * 0.2),  # テストサイズを20%に固定
            gap=int(len(X) * 0.02)  # ギャップを2%に調整
        )
        
        # モデルのパラメータ（改善版）
        params = {
            'objective': 'multiclass',
            'metric': ['multi_logloss', 'multi_error'],  # 複数の評価指標を追加
            'num_class': 3,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,  # オーバーフィッティング防止
            'max_depth': 6,          # 深さの制限
            'verbose': -1
        }
        
        # モデルの評価結果を保存
        all_metrics = []
        feature_importance_gain = np.zeros(len(X.columns))
        best_model = None
        best_score = 0
        
        logger.info("モデルのトレーニングを開始します...")
        
        # 時系列交差検証
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # クラスの重みを計算
            class_counts = np.bincount(y_train)
            class_weights = dict(zip(
                range(3),  # 0, 1, 2のクラスラベル
                len(y_train) / (3 * class_counts)
            ))
            
            # トレーニングデータセットの作成
            train_data = lgb.Dataset(X_train, label=y_train, weight=np.array([class_weights[y] for y in y_train]))
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # モデルのトレーニング
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # 予測と元のラベルへの変換
            y_pred = np.argmax(model.predict(X_val), axis=1)
            
            # 評価用に予測値と実際の値を元のラベル（-1, 0, 1）に戻す
            reverse_mapping = {0: -1, 1: 0, 2: 1}
            y_val_original = y_val.map(reverse_mapping)
            y_pred_original = pd.Series(y_pred).map(reverse_mapping)
            
            # 評価メトリクスの計算
            report = classification_report(y_val_original, y_pred_original, output_dict=True)
            all_metrics.append(report)
            
            # 特徴量重要度の累積
            feature_importance_gain += model.feature_importance(importance_type='gain')
            
            # 最良モデルの保存
            current_score = report['weighted avg']['f1-score']
            if best_model is None or current_score > best_score:
                best_model = model
                best_score = current_score
            
            logger.info(f"\nFold {fold} の結果:")
            logger.info(f"Accuracy: {report['accuracy']:.4f}")
            logger.info(f"Precision: {report['weighted avg']['precision']:.4f}")
            logger.info(f"Recall: {report['weighted avg']['recall']:.4f}")
            logger.info(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
            
            # 混同行列の表示と保存
            cm = confusion_matrix(y_val_original, y_pred_original)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Fold {fold} の混同行列')
            plt.xlabel('予測値')
            plt.ylabel('実際の値')
            plt.savefig(f"reports/confusion_matrix_fold{fold}_{datetime.now().strftime('%Y%m%d')}.png")
            plt.close()
            
            logger.info("\n混同行列:")
            logger.info(cm)
        
        # 平均特徴量重要度の計算
        feature_importance_gain /= fold
        
        # 最良モデルの保存
        os.makedirs("models", exist_ok=True)
        model_path = f"models/model_{datetime.now().strftime('%Y%m%d')}.txt"
        best_model.save_model(model_path)
        
        # 特徴量重要度の可視化と保存
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance_gain
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('特徴量重要度（平均ゲイン）')
        plt.tight_layout()
        
        os.makedirs("reports", exist_ok=True)
        plt.savefig(f"reports/feature_importance_{datetime.now().strftime('%Y%m%d')}.png")
        
        # 評価メトリクスの平均を計算
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
            'precision': np.mean([m['weighted avg']['precision'] for m in all_metrics]),
            'recall': np.mean([m['weighted avg']['recall'] for m in all_metrics]),
            'f1_score': np.mean([m['weighted avg']['f1-score'] for m in all_metrics])
        }
        
        logger.info("\n平均評価メトリクス:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # 評価結果をJSONファイルとして保存
        results = {
            'average_metrics': avg_metrics,
            'feature_importance': importance_df.to_dict(),
            'model_parameters': params,
            'training_date': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        with open(f"reports/training_results_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        return True
        
    except Exception as e:
        logger.error(f"モデルトレーニング中にエラーが発生しました: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    print("機械学習モデルの学習を開始します...")
    
    # データの読み込みと前処理
    df = pd.read_csv('./data/features_20250217.csv')
    df = preprocess_data(df)
    
    print(f"データ準備完了 (行数: {len(df)}, 特徴量数: {df.shape[1]})")
    
    # モデルの学習
    results = train_all_models(df, STRATEGY_PATTERNS)
    
    # 結果の保存
    with open('./results/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 成功したモデルの数を表示
    successful_models = len(results)
    total_models = len(STRATEGY_PATTERNS)
    print(f"\n学習完了: {successful_models}/{total_models} モデルが正常に学習されました")

if __name__ == "__main__":
    train_model() 