"""
機械学習モデルの学習スクリプト
各種モデルのトレーニングと保存を行う
"""

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
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """データの準備"""
        # 特徴量とターゲットの分離
        X = self.df[self.strategy.features].values
        y = self.df[self.strategy.prediction_target].values
        
        # スケーリング
        if self.strategy.preprocessing['scaler'] == 'standard':
            self.feature_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
        
        X = self.feature_scaler.fit_transform(X)
        
        # ターゲットのスケーリング（回帰問題の場合）
        if self.strategy.prediction_target in [
            PredictionTargets.PRICE_CHANGE,
            PredictionTargets.NEXT_PRICE,
            PredictionTargets.OPTIMAL_POSITION,
            PredictionTargets.RISK_REWARD
        ]:
            self.scaler = MinMaxScaler()
            y = self.scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        return X, y
    
    def prepare_lstm_data(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """LSTMのための時系列データ準備"""
        X_seq = []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:(i + sequence_length)])
        return np.array(X_seq)
    
    def train_model(self) -> Tuple[object, Dict]:
        """モデルの学習"""
        # データの準備
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # モデルの選択と学習
        if self.strategy.model_type == 'random_forest':
            if self.is_classification_task():
                self.model = MLModels.get_random_forest_classifier()
            else:
                self.model = MLModels.get_random_forest_regressor()
                
        elif self.strategy.model_type == 'xgboost':
            if self.is_classification_task():
                self.model = MLModels.get_xgboost_classifier()
            else:
                self.model = MLModels.get_xgboost_regressor()
                
        elif self.strategy.model_type == 'lstm':
            # LSTMデータの準備
            X_train = self.prepare_lstm_data(X_train)
            X_test = self.prepare_lstm_data(X_test)
            y_train = y_train[10:]  # sequence_lengthに合わせる
            y_test = y_test[10:]
            
            self.model = MLModels.get_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                output_shape=1
            )
        
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
        model_path = f"{base_path}/{self.strategy.name}"
        
        # モデルの保存
        if self.strategy.model_type == 'lstm':
            self.model.save(f"{model_path}_model")
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
    
    for strategy in strategies:
        print(f"\n学習中: {strategy.name}")
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
            
            print(f"完了: {strategy.name}")
            print(f"メトリクス: {metrics}")
            
        except Exception as e:
            print(f"エラー ({strategy.name}): {str(e)}")
            continue
    
    return results

def main():
    # データの読み込み
    df = pd.read_csv('./data/features_20250217.csv')
    
    # すべてのモデルを学習
    results = train_all_models(df, STRATEGY_PATTERNS)
    
    # 結果の保存
    with open('./results/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n学習が完了しました。")
    print(f"学習したモデル数: {len(results)}")

if __name__ == "__main__":
    main() 