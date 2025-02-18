"""
機械学習モデルを統合した取引戦略
既存の技術的分析と機械学習モデルの予測を組み合わせて取引判断を行う
"""

import numpy as np
from typing import List, Dict, Tuple
import joblib
import json
from datetime import datetime
from p19_strategy import TradingStrategy
from ml_strategy_patterns import PredictionTargets
import tensorflow as tf

class MLTradingStrategy(TradingStrategy):
    def __init__(self):
        """初期化"""
        super().__init__()
        self.models = {}
        self.scalers = {}
        self.configs = {}
        self.load_models()
        
        # モデルの予測重み
        self.model_weights = {
            'price_direction': 0.3,    # 価格方向の予測
            'trend_strength': 0.2,     # トレンド強度の予測
            'trading_signal': 0.3,     # 取引シグナルの予測
            'optimal_position': 0.2    # ポジションサイズの予測
        }
    
    def load_models(self):
        """最適なモデルの読み込み"""
        # 評価レポートの読み込み
        with open('./results/evaluation_report.json', 'r') as f:
            evaluation = json.load(f)
        
        # 最適なモデルの読み込み
        for target, model_name in evaluation['best_models'].items():
            base_path = f"./models/{model_name}"
            
            # 設定の読み込み
            with open(f"{base_path}_config.json", 'r') as f:
                self.configs[target] = json.load(f)
            
            # モデルの読み込み
            if self.configs[target]['model_type'] == 'lstm':
                self.models[target] = tf.keras.models.load_model(f"{base_path}_model")
            else:
                self.models[target] = joblib.load(f"{base_path}_model.joblib")
            
            # スケーラーの読み込み
            self.scalers[target] = {
                'feature': joblib.load(f"{base_path}_feature_scaler.joblib")
            }
            if target in [PredictionTargets.PRICE_CHANGE, PredictionTargets.NEXT_PRICE,
                         PredictionTargets.OPTIMAL_POSITION, PredictionTargets.RISK_REWARD]:
                self.scalers[target]['target'] = joblib.load(f"{base_path}_target_scaler.joblib")
    
    def prepare_features(self, price_history: List[float], current_time: datetime) -> Dict[str, np.ndarray]:
        """特徴量の準備"""
        # 基本的な指標の計算
        trend_score, volatility, price_change = super().calculate_trend(price_history, current_time)
        
        # 各モデルの特徴量を準備
        features = {}
        for target, config in self.configs.items():
            feature_values = []
            for feature in config['features']:
                if feature == 'rsi_14':
                    value = self.current_rsi
                elif feature == 'volatility':
                    value = volatility
                elif feature == 'trend_strength':
                    value = abs(trend_score)
                elif feature == 'price_momentum':
                    value = price_change
                elif feature == 'hour_of_day':
                    value = current_time.hour
                elif feature == 'day_of_week':
                    value = current_time.weekday()
                else:
                    value = 0  # デフォルト値
                
                feature_values.append(value)
            
            # スケーリング
            X = np.array(feature_values).reshape(1, -1)
            X = self.scalers[target]['feature'].transform(X)
            features[target] = X
        
        return features
    
    def get_ml_predictions(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """機械学習モデルによる予測"""
        predictions = {}
        
        for target, X in features.items():
            if target not in self.models:
                continue
            
            # 予測
            pred = self.models[target].predict(X)
            
            # スケーリング逆変換（回帰モデルの場合）
            if target in [PredictionTargets.PRICE_CHANGE, PredictionTargets.NEXT_PRICE,
                         PredictionTargets.OPTIMAL_POSITION, PredictionTargets.RISK_REWARD]:
                pred = self.scalers[target]['target'].inverse_transform(pred.reshape(-1, 1)).ravel()
            
            predictions[target] = float(pred[0])
        
        return predictions
    
    def calculate_ml_trade_signal(self, predictions: Dict[str, float]) -> float:
        """機械学習モデルの予測から取引シグナルを計算"""
        signal = 0.0
        
        # 価格方向の予測
        if 'price_direction' in predictions:
            signal += (predictions['price_direction'] - 0.5) * 2 * self.model_weights['price_direction']
        
        # トレンド強度の予測
        if 'trend_strength' in predictions:
            signal += (predictions['trend_strength'] - 1) * self.model_weights['trend_strength']
        
        # 取引シグナルの予測
        if 'trading_signal' in predictions:
            signal += predictions['trading_signal'] * self.model_weights['trading_signal']
        
        return signal
    
    def should_buy(self, trend_score: float, volatility: float, current_time: datetime = None) -> bool:
        """買いシグナルの判定（機械学習モデルの予測を考慮）"""
        # 基本的な条件チェック
        if not super().should_buy(trend_score, volatility, current_time):
            return False
        
        # 機械学習モデルの予測を取得
        features = self.prepare_features(self.trend_memory, current_time)
        predictions = self.get_ml_predictions(features)
        
        # 機械学習シグナルの計算
        ml_signal = self.calculate_ml_trade_signal(predictions)
        
        # 機械学習モデルの予測も買いを示している場合のみTrue
        return ml_signal < -0.3
    
    def should_sell(self, trend_score: float, current_time: datetime = None) -> bool:
        """売りシグナルの判定（機械学習モデルの予測を考慮）"""
        # 基本的な条件チェック
        if not super().should_sell(trend_score, current_time):
            return False
        
        # 機械学習モデルの予測を取得
        features = self.prepare_features(self.trend_memory, current_time)
        predictions = self.get_ml_predictions(features)
        
        # 機械学習シグナルの計算
        ml_signal = self.calculate_ml_trade_signal(predictions)
        
        # 機械学習モデルの予測も売りを示している場合のみTrue
        return ml_signal > 0.3
    
    def calculate_optimal_trade_amount(self, current_price: float, trend_score: float,
                                     volatility: float, available_balance: float,
                                     current_time: datetime = None) -> float:
        """最適な取引量を計算（機械学習モデルの予測を考慮）"""
        # 基本的な取引量の計算
        base_amount = super().calculate_optimal_trade_amount(
            current_price, trend_score, volatility, available_balance, current_time
        )
        
        # 機械学習モデルの予測を取得
        features = self.prepare_features(self.trend_memory, current_time)
        predictions = self.get_ml_predictions(features)
        
        # ポジションサイズの予測がある場合は考慮
        if 'optimal_position' in predictions:
            ml_position_size = predictions['optimal_position']
            # 基本取引量とMLの予測を組み合わせる
            final_amount = base_amount * (0.7 + 0.3 * ml_position_size)
            return min(final_amount, available_balance)
        
        return base_amount 