"""
機械学習戦略のパターン集

このファイルでは、以下の要素を組み合わせた様々な機械学習戦略を定義します：
1. 予測目的
2. 特徴量
3. 機械学習アルゴリズム
4. 予測時間枠
5. データの前処理方法
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 予測目標の定義
class PredictionTargets:
    # 分類問題
    PRICE_DIRECTION = "price_direction"       # 価格の上昇/下降を予測
    TREND_STRENGTH = "trend_strength"         # トレンドの強さを予測
    TRADING_SIGNAL = "trading_signal"         # 取引シグナル（買い/売り/待機）を予測
    VOLATILITY_LEVEL = "volatility_level"     # ボラティリティレベルを予測
    
    # 回帰問題
    PRICE_CHANGE = "price_change"            # 価格変化率を予測
    NEXT_PRICE = "next_price"                # 次の価格を予測
    OPTIMAL_POSITION = "optimal_position"     # 最適なポジションサイズを予測
    RISK_REWARD = "risk_reward"              # リスク/リワード比を予測

# 特徴量パターンの定義
class FeaturePatterns:
    @staticmethod
    def get_basic_features() -> List[str]:
        """基本的な特徴量"""
        return [
            "rsi_14",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "ma_5",
            "ma_13",
            "ma_21",
            "volatility"
        ]
    
    @staticmethod
    def get_advanced_features() -> List[str]:
        """より高度な特徴量"""
        return FeaturePatterns.get_basic_features() + [
            "macd",
            "macd_signal",
            "macd_hist",
            "stoch_k",
            "stoch_d",
            "adx",
            "cci",
            "obv",
            "atr"
        ]
    
    @staticmethod
    def get_price_action_features() -> List[str]:
        """価格アクション関連の特徴量"""
        return [
            "body_size",          # ローソク足の実体の大きさ
            "upper_shadow",       # 上ヒゲの長さ
            "lower_shadow",       # 下ヒゲの長さ
            "price_momentum",     # 価格モメンタム
            "volume_momentum",    # 出来高モメンタム
            "price_acceleration", # 価格の加速度
            "trend_strength"      # トレンドの強さ
        ]
    
    @staticmethod
    def get_market_timing_features() -> List[str]:
        """市場タイミング関連の特徴量"""
        return [
            "hour_of_day",        # 時間帯
            "day_of_week",        # 曜日
            "volatility_regime",   # ボラティリティレジーム
            "trend_regime",        # トレンドレジーム
            "market_phase"         # 市場フェーズ
        ]

# 機械学習モデルのパターン
class MLModels:
    @staticmethod
    def get_random_forest_classifier(n_estimators=100):
        """ランダムフォレスト分類器"""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    @staticmethod
    def get_random_forest_regressor(n_estimators=100):
        """ランダムフォレスト回帰器"""
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    @staticmethod
    def get_xgboost_classifier():
        """XGBoost分類器"""
        return XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    @staticmethod
    def get_xgboost_regressor():
        """XGBoost回帰器"""
        return XGBRegressor(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    @staticmethod
    def get_lstm_model(input_shape, output_shape):
        """LSTMモデル"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(output_shape)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

# 予測時間枠のパターン
class TimeFrames:
    NEXT_MINUTE = 1        # 1分後を予測
    NEXT_5_MINUTES = 5     # 5分後を予測
    NEXT_15_MINUTES = 15   # 15分後を予測
    NEXT_30_MINUTES = 30   # 30分後を予測
    NEXT_HOUR = 60         # 1時間後を予測

# 戦略パターンの定義
@dataclass
class StrategyPattern:
    name: str                      # 戦略名
    prediction_target: str         # 予測目標
    features: List[str]            # 使用する特徴量
    model_type: str               # モデルの種類
    time_frame: int               # 予測時間枠
    preprocessing: Dict           # 前処理パラメータ

# 戦略パターンの例
STRATEGY_PATTERNS = [
    StrategyPattern(
        name="basic_direction_rf",
        prediction_target=PredictionTargets.PRICE_DIRECTION,
        features=FeaturePatterns.get_basic_features(),
        model_type="random_forest_classifier",
        time_frame=TimeFrames.NEXT_5_MINUTES,
        preprocessing={"scaler": "standard"}
    ),
    
    StrategyPattern(
        name="advanced_trend_xgb",
        prediction_target=PredictionTargets.TREND_STRENGTH,
        features=FeaturePatterns.get_advanced_features(),
        model_type="xgboost_classifier",
        time_frame=TimeFrames.NEXT_15_MINUTES,
        preprocessing={"scaler": "standard"}
    ),
    
    StrategyPattern(
        name="price_action_lstm",
        prediction_target=PredictionTargets.NEXT_PRICE,
        features=FeaturePatterns.get_price_action_features(),
        model_type="lstm",
        time_frame=TimeFrames.NEXT_MINUTE,
        preprocessing={"scaler": "minmax"}
    ),
    
    StrategyPattern(
        name="market_timing_rf",
        prediction_target=PredictionTargets.TRADING_SIGNAL,
        features=FeaturePatterns.get_market_timing_features(),
        model_type="random_forest_classifier",
        time_frame=TimeFrames.NEXT_30_MINUTES,
        preprocessing={"scaler": "standard"}
    ),
    
    StrategyPattern(
        name="volatility_xgb",
        prediction_target=PredictionTargets.VOLATILITY_LEVEL,
        features=FeaturePatterns.get_advanced_features() + FeaturePatterns.get_market_timing_features(),
        model_type="xgboost_classifier",
        time_frame=TimeFrames.NEXT_HOUR,
        preprocessing={"scaler": "robust"}
    ),
    
    StrategyPattern(
        name="position_size_lstm",
        prediction_target=PredictionTargets.OPTIMAL_POSITION,
        features=FeaturePatterns.get_advanced_features() + FeaturePatterns.get_price_action_features(),
        model_type="lstm",
        time_frame=TimeFrames.NEXT_5_MINUTES,
        preprocessing={"scaler": "minmax"}
    )
]

def create_feature_combinations():
    """特徴量の組み合わせを生成"""
    basic = set(FeaturePatterns.get_basic_features())
    advanced = set(FeaturePatterns.get_advanced_features())
    price_action = set(FeaturePatterns.get_price_action_features())
    market_timing = set(FeaturePatterns.get_market_timing_features())
    
    combinations = [
        basic,
        advanced,
        basic.union(price_action),
        advanced.union(price_action),
        basic.union(market_timing),
        advanced.union(market_timing),
        basic.union(price_action).union(market_timing),
        advanced.union(price_action).union(market_timing)
    ]
    
    return combinations

def generate_strategy_patterns():
    """戦略パターンを生成"""
    patterns = []
    feature_combinations = create_feature_combinations()
    prediction_targets = [
        PredictionTargets.PRICE_DIRECTION,
        PredictionTargets.TREND_STRENGTH,
        PredictionTargets.TRADING_SIGNAL,
        PredictionTargets.NEXT_PRICE
    ]
    time_frames = [
        TimeFrames.NEXT_MINUTE,
        TimeFrames.NEXT_5_MINUTES,
        TimeFrames.NEXT_15_MINUTES
    ]
    
    for features in feature_combinations:
        for target in prediction_targets:
            for time_frame in time_frames:
                # ランダムフォレストパターン
                patterns.append(StrategyPattern(
                    name=f"rf_{target}_{time_frame}min",
                    prediction_target=target,
                    features=list(features),
                    model_type="random_forest",
                    time_frame=time_frame,
                    preprocessing={"scaler": "standard"}
                ))
                
                # XGBoostパターン
                patterns.append(StrategyPattern(
                    name=f"xgb_{target}_{time_frame}min",
                    prediction_target=target,
                    features=list(features),
                    model_type="xgboost",
                    time_frame=time_frame,
                    preprocessing={"scaler": "standard"}
                ))
                
                # LSTMパターン（時系列データに特化）
                if len(features) > 5:  # 特徴量が十分にある場合のみ
                    patterns.append(StrategyPattern(
                        name=f"lstm_{target}_{time_frame}min",
                        prediction_target=target,
                        features=list(features),
                        model_type="lstm",
                        time_frame=time_frame,
                        preprocessing={"scaler": "minmax"}
                    ))
    
    return patterns

# 使用例
if __name__ == "__main__":
    # すべての戦略パターンを生成
    all_patterns = generate_strategy_patterns()
    print(f"生成された戦略パターン数: {len(all_patterns)}")
    
    # パターンの例を表示
    for i, pattern in enumerate(all_patterns[:5]):
        print(f"\nパターン {i+1}:")
        print(f"名前: {pattern.name}")
        print(f"予測目標: {pattern.prediction_target}")
        print(f"特徴量数: {len(pattern.features)}")
        print(f"モデル種類: {pattern.model_type}")
        print(f"時間枠: {pattern.time_frame}分") 