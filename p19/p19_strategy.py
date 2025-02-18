"""
取引戦略のコアロジックを含むクラス
RSI、ボリンジャーバンド、移動平均線を組み合わせた総合的な分析
トレンド分析の強化
取引シグナルの精度向上
ポジションサイズの最適化
損切り・利確の判断
"""
import numpy as np
from typing import List, Tuple
import math
from datetime import datetime
import lightgbm as lgb
import pandas as pd
import os

class TradingStrategy:
    def __init__(self):
        self.rsi_period = 14
        self.rsi_oversold = 30        # RSI過売り閾値を調整
        self.rsi_overbought = 70      # RSI過買い閾値を調整
        self.stop_loss_pct = 0.015    # ストップロスを1.5%に調整
        self.take_profit_pct = 0.03   # 利確を3%に調整
        self.min_trade_amount = 0.001  # 最小取引量
        self.max_trade_amount = 0.01   # 最大取引量
        self.position_size_pct = 0.1   # ポジションサイズを10%に増加
        self.history_size = 90
        self.ma_short = 5
        self.ma_mid = 13
        self.ma_long = 21
        self.current_rsi = 50.0
        self.bb_period = 20
        self.bb_std = 2.0
        self.trend_memory = []
        self.trend_memory_size = 5
        self.last_trade_price = None
        self.last_trade_time = None
        self.min_trade_interval = 1800  # 取引間隔を30分に短縮
        
        # モデルの読み込み
        self.load_ml_model()
        
        # 収益の高い時間帯に最適化
        self.preferred_hours = {8, 14, 18, 22}  # 収益の高い時間帯を追加
        self.avoid_hours = {5, 6, 17}          # 損失の大きい時間帯

    def load_ml_model(self):
        """機械学習モデルの読み込み"""
        models_dir = "models"
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.txt')]
        if not model_files:
            raise FileNotFoundError("モデルファイルが見つかりません")
        
        latest_model = max(model_files)
        self.model = lgb.Booster(model_file=os.path.join(models_dir, latest_model))

    def prepare_features(self, price_history: List[float]) -> pd.DataFrame:
        """機械学習モデル用の特徴量を準備"""
        # 1行のデータフレームを作成
        df = pd.DataFrame({
            'open': [price_history[-1]],
            'high': [max(price_history[-20:])],
            'low': [min(price_history[-20:])],
            'close': [price_history[-1]],
            'volume': [np.mean(price_history[-5:])]
        })

        # RSI
        df['rsi'] = [self.calculate_rsi(price_history)]

        # ボリンジャーバンド
        upper, middle, lower = self.calculate_bollinger_bands(price_history)
        df['bollinger_high'] = [upper]
        df['bollinger_low'] = [lower]

        # MACD（簡易版）
        df['macd'] = [np.mean(price_history[-12:]) - np.mean(price_history[-26:])]
        df['macd_signal'] = [np.mean(price_history[-9:])]

        # ATR（簡易版）
        df['atr'] = [np.std(price_history[-14:])]

        # リターン
        df['returns'] = [(price_history[-1] / price_history[-2]) - 1 if len(price_history) > 1 else 0]

        # モメンタム
        df['momentum_1'] = [(price_history[-1] / price_history[-2]) - 1 if len(price_history) > 1 else 0]
        df['momentum_5'] = [(price_history[-1] / price_history[-5]) - 1 if len(price_history) > 5 else 0]
        df['momentum_10'] = [(price_history[-1] / price_history[-10]) - 1 if len(price_history) > 10 else 0]

        # ボラティリティ
        returns = [price_history[i]/price_history[i-1]-1 for i in range(-5,0)]
        df['volatility_5'] = [np.std(returns)]
        returns = [price_history[i]/price_history[i-1]-1 for i in range(-10,0)]
        df['volatility_10'] = [np.std(returns)]
        
        vol_5 = df['volatility_5'].iloc[0]
        vol_10 = df['volatility_10'].iloc[0]
        df['volatility_ratio'] = [vol_5 / vol_10 if vol_10 != 0 else 1]

        # トレンド強度
        df['trend_strength'] = [abs(df['momentum_5'].iloc[0])]

        # 価格レンジ
        price_range = df['high'].iloc[0] - df['low'].iloc[0]
        df['price_range_ratio'] = [price_range / df['close'].iloc[0] if df['close'].iloc[0] != 0 else 0]
        df['body_ratio'] = [abs(df['close'].iloc[0] - df['open'].iloc[0]) / price_range if price_range != 0 else 0]
        df['relative_volume'] = [df['volume'].iloc[0] / np.mean(price_history[-5:]) if np.mean(price_history[-5:]) != 0 else 1]

        # 特徴量の順序を学習時と同じにする
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'bollinger_high', 'bollinger_low',
            'macd', 'macd_signal', 'atr',
            'returns', 'momentum_1', 'momentum_5', 'momentum_10',
            'volatility_5', 'volatility_10', 'trend_strength',
            'price_range_ratio', 'body_ratio', 'relative_volume'
        ]
        
        return df[feature_columns]

    def get_ml_prediction(self, price_history: List[float]) -> float:
        """機械学習モデルによる予測"""
        if len(price_history) < self.history_size:
            return 0.5

        features = self.prepare_features(price_history)
        prediction = self.model.predict(features)
        
        # 3クラス分類の結果を確率に変換
        probabilities = prediction[0]
        weighted_prob = (probabilities[2] * 1.0 + probabilities[1] * 0.5 + probabilities[0] * 0.0)
        
        return weighted_prob

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI（Relative Strength Index）を計算"""
        if len(prices) < period + 1:
            return 50.0  # データが不足している場合は中立値を返す
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20) -> Tuple[float, float, float]:
        """ボリンジャーバンドを計算"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        prices_array = np.array(prices[-period:])
        sma = np.mean(prices_array)
        std = np.std(prices_array)
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        return upper_band, sma, lower_band

    def calculate_trend(self, price_history):
        """トレンドとボラティリティを計算（改善版）"""
        if len(price_history) < 20:  # 最低20個のデータポイントが必要
            return 0, 0
        
        # 価格変化率を計算（より短期的な変化に注目）
        recent_prices = np.array(price_history[-10:])
        price_changes = np.diff(recent_prices) / recent_prices[:-1]
        
        # 移動平均を計算
        ma_short = np.mean(price_history[-5:])
        ma_mid = np.mean(price_history[-10:])
        ma_long = np.mean(price_history[-20:])
        
        # トレンドスコアの計算（改善版）
        trend_score = 0
        
        # 短期トレンド（50%のウェイト）
        short_trend = (ma_short / ma_mid - 1) * 5
        trend_score += short_trend * 0.5
        
        # 中期トレンド（30%のウェイト）
        mid_trend = (ma_mid / ma_long - 1) * 3
        trend_score += mid_trend * 0.3
        
        # 直近の価格変化（20%のウェイト）
        recent_change = np.mean(price_changes) * 10
        trend_score += recent_change * 0.2
        
        # ボラティリティの計算（改善版）
        volatility = np.std(price_changes) * np.sqrt(len(price_changes))
        
        # トレンドメモリを更新
        self.trend_memory.append(trend_score)
        if len(self.trend_memory) > self.trend_memory_size:
            self.trend_memory.pop(0)
        
        return trend_score, volatility

    def is_good_trading_time(self, current_time: datetime) -> bool:
        """取引に適した時間帯かどうかを判断"""
        hour = current_time.hour
        
        # 避けるべき時間帯の場合
        if hour in self.avoid_hours:
            return False
            
        # 好ましい時間帯の場合、より緩和された条件で取引
        if hour in self.preferred_hours:
            return True
            
        # その他の時間帯は通常の条件で取引
        return True

    def should_close_position(self, current_price: float, entry_price: float) -> bool:
        """ポジションクローズの判断"""
        if entry_price is None:
            return False
        
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        
        # 損切り条件
        if price_change_pct < -self.stop_loss_pct:
            return True
        
        # 利確条件
        if price_change_pct > self.take_profit_pct:
            return True
        
        return False

    def calculate_optimal_trade_amount(self, current_price: float, trend_score: float, volatility: float, available_balance: float, current_time: datetime = None) -> float:
        """最適な取引量を計算（改善版）"""
        if available_balance < self.min_trade_amount:
            return 0
        
        # 市場状態に基づく基本取引率の決定
        trend_strength = abs(trend_score)
        market_condition = self.analyze_market_condition(trend_score, volatility)
        
        if market_condition == 'strong':
            base_ratio = 0.15  # より積極的な比率
        elif market_condition == 'moderate':
            base_ratio = 0.1   # 標準的な比率
        else:
            base_ratio = 0.05  # より控えめな比率
        
        # 好ましい時間帯では取引サイズを増加
        if current_time:
            if current_time.hour in self.preferred_hours:
                base_ratio *= 1.2
            elif current_time.hour in self.avoid_hours:
                base_ratio *= 0.5
        
        # トレンドの一貫性による調整
        if len(self.trend_memory) >= self.trend_memory_size:
            trend_std = np.std(self.trend_memory)
            consistency_factor = 1.0 - min(trend_std, 0.3)
        else:
            consistency_factor = 0.5
        
        # ボラティリティによる調整（改善版）
        if volatility < 0.1:
            volatility_factor = 0.8  # 低ボラティリティでは控えめに
        elif volatility < 0.3:
            volatility_factor = 1.0  # 適度なボラティリティでは通常通り
        else:
            volatility_factor = 0.7  # 高ボラティリティでは控えめに
        
        # RSIの極値による調整（改善版）
        rsi_factor = 1.0
        if self.current_rsi < 20 or self.current_rsi > 80:
            rsi_factor = 1.3  # より積極的に
        elif self.current_rsi < 30 or self.current_rsi > 70:
            rsi_factor = 1.1  # やや積極的に
        
        final_ratio = base_ratio * consistency_factor * volatility_factor * rsi_factor
        
        # 最終的な取引量の計算
        trade_amount = min(available_balance * final_ratio, self.max_trade_amount)
        trade_amount = max(trade_amount, self.min_trade_amount)
        
        return min(trade_amount, available_balance)

    def analyze_market_condition(self, trend_score: float, volatility: float) -> str:
        """市場状態の分析"""
        trend_strength = abs(trend_score)
        
        if trend_strength > 0.3 and volatility > 0.2:
            return 'strong'    # 強いトレンドと適度なボラティリティ
        elif trend_strength > 0.15 and volatility > 0.1:
            return 'moderate'  # 中程度のトレンドとボラティリティ
        else:
            return 'weak'      # 弱いトレンドまたは低ボラティリティ

    def should_buy(self, trend_score, avg_trend, recent_trend, volatility, current_time=None):
        """買いシグナルの判定（改善版）"""
        if len(self.trend_memory) < self.trend_memory_size:
            return False
        
        # 取引を避ける時間帯かチェック
        if current_time and isinstance(current_time, datetime) and current_time.hour in self.avoid_hours:
            return False
        
        # 市場状態の確認
        market_condition = self.analyze_market_condition(trend_score, volatility)
        
        # 市場状態に応じた条件設定
        if market_condition == 'strong':
            # より積極的な条件
            if trend_score > -0.1:
                return False
            if avg_trend > -0.05:
                return False
            if recent_trend > -0.15:
                return False
        else:
            # より慎重な条件
            if trend_score > -0.15:
                return False
            if avg_trend > -0.1:
                return False
            if recent_trend > -0.2:
                return False
        
        # ボラティリティの条件をチェック
        if volatility < 0.05 or volatility > 0.8:
            return False
        
        # RSIの条件をチェック（市場状態に応じて調整）
        if market_condition == 'strong':
            if self.current_rsi > 40:  # より積極的に
                return False
        else:
            if self.current_rsi > 30:  # より慎重に
                return False
        
        return True

    def should_sell(self, trend_score, avg_trend, recent_trend, volatility, current_time=None):
        """売りシグナルの判定（改善版）"""
        if len(self.trend_memory) < self.trend_memory_size:
            return False
        
        # 取引を避ける時間帯かチェック
        if current_time and isinstance(current_time, datetime) and current_time.hour in self.avoid_hours:
            return False
        
        # 市場状態の確認
        market_condition = self.analyze_market_condition(trend_score, volatility)
        
        # 市場状態に応じた条件設定
        if market_condition == 'strong':
            # より積極的な条件
            if trend_score < 0.1:
                return False
            if avg_trend < 0.05:
                return False
            if recent_trend < 0.15:
                return False
        else:
            # より慎重な条件
            if trend_score < 0.15:
                return False
            if avg_trend < 0.1:
                return False
            if recent_trend < 0.2:
                return False
        
        # ボラティリティの条件をチェック
        if volatility < 0.05 or volatility > 0.8:
            return False
        
        # RSIの条件をチェック（市場状態に応じて調整）
        if market_condition == 'strong':
            if self.current_rsi < 60:  # より積極的に
                return False
        else:
            if self.current_rsi < 70:  # より慎重に
                return False
        
        return True

    def calculate_position_size(self, price: float, amount: float) -> float:
        """ポジションサイズの計算（BTCの数量を5桁に切り上げ）"""
        btc_qty = amount / price
        return math.ceil(btc_qty * 100000) / 100000 