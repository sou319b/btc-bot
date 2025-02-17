"""
機械学習モデルの学習スクリプト
各種モデルのトレーニングと保存を行う
"""

import os
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
    main() 