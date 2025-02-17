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

class TradingStrategy:
    def __init__(self):
        self.rsi_period = 14
        self.rsi_oversold = 25      # RSIをより厳格に
        self.rsi_overbought = 75
        self.stop_loss_pct = 0.2    # 損切りを調整
        self.take_profit_pct = 0.8  # 利確ラインを高く
        self.min_trade_amount = 15   # 最小取引額を増加
        self.max_trade_amount = 35   # 最大取引額を増加
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
        self.min_trade_interval = 3600  # 1時間に短縮
        
        # 収益の高い時間帯に最適化（バックテスト結果から）
        self.preferred_hours = {8, 14, 18}  # 収益の高い時間帯
        self.avoid_hours = {5, 6, 17}      # 損失の大きい時間帯

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

    def calculate_trend(self, price_history: List[float], current_time: datetime = None) -> Tuple[float, float, float]:
        """価格トレンドとボラティリティを計算"""
        if len(price_history) < self.history_size:
            return 0, 0, 0
        
        self.current_rsi = self.calculate_rsi(price_history, self.rsi_period)
        upper, middle, lower = self.calculate_bollinger_bands(price_history, self.bb_period)
        
        # 移動平均の計算
        ma_short = np.mean(price_history[-self.ma_short:])
        ma_mid = np.mean(price_history[-self.ma_mid:])
        ma_long = np.mean(price_history[-self.ma_long:])
        
        current_price = price_history[-1]
        band_width = upper - lower
        volatility = (current_price - lower) / band_width if band_width != 0 else 0
        
        try:
            # 価格変化の計算
            price_changes = []
            for period in [3, 5, 8, 13]:
                if len(price_history) > period:
                    change = (price_history[-1] - price_history[-period]) / price_history[-period] * 100
                    price_changes.append(change)
                else:
                    price_changes.append(0)
            
            # 移動平均のアライメント分析
            ma_alignment = 0
            if ma_short > ma_mid > ma_long and current_price > ma_short * 1.002:
                ma_alignment = 2
            elif ma_short < ma_mid < ma_long and current_price < ma_short * 0.998:
                ma_alignment = -2
            
            # ボリンジャーバンドの分析
            bb_score = 0
            if current_price <= lower * 0.997:
                bb_score = -2
            elif current_price >= upper * 1.003:
                bb_score = 2
            else:
                bb_score = (current_price - middle) / (upper - middle) if upper != middle else 0
            
            # トレンドスコアの計算
            trend_score = (
                ((self.current_rsi - 50) / 25) * 0.25 +
                np.mean(price_changes) * 0.35 +
                ma_alignment * 0.25 +
                bb_score * 0.15
            )
            
            # 時間帯による調整
            if current_time:
                if current_time.hour in self.preferred_hours:
                    trend_score *= 1.5  # 好ましい時間帯での信号を強調
                elif current_time.hour in self.avoid_hours:
                    trend_score *= 0.3  # 避けるべき時間帯での信号を大幅に弱める
            
            self.trend_memory.append(trend_score)
            if len(self.trend_memory) > self.trend_memory_size:
                self.trend_memory.pop(0)
            
            return trend_score, volatility, price_changes[0]
        except (IndexError, ZeroDivisionError):
            return 0, volatility, 0

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
        """最適な取引量を計算"""
        if available_balance < self.min_trade_amount:
            return 0
        
        # 基本取引率の決定
        trend_strength = abs(trend_score)
        if trend_strength > 0.6:
            base_ratio = 0.4  # より積極的な比率
        elif trend_strength > 0.4:
            base_ratio = 0.3
        else:
            base_ratio = 0.2
        
        # 好ましい時間帯では取引サイズを増加
        if current_time and current_time.hour in self.preferred_hours:
            base_ratio *= 1.3
        elif current_time and current_time.hour in self.avoid_hours:
            base_ratio *= 0.5
        
        # トレンドの一貫性による調整
        if len(self.trend_memory) >= self.trend_memory_size:
            trend_std = np.std(self.trend_memory)
            consistency_factor = 1.0 - min(trend_std, 0.3)  # より積極的な一貫性
        else:
            consistency_factor = 0.5
        
        # ボラティリティによる調整
        volatility_factor = 0.9 + (0.2 * (1.0 - min(volatility, 0.3)))
        
        # RSIの極値による調整
        rsi_factor = 1.0
        if (self.current_rsi < 25 and trend_score < 0) or (self.current_rsi > 75 and trend_score > 0):
            rsi_factor = 1.2
        
        final_ratio = base_ratio * consistency_factor * volatility_factor * rsi_factor
        
        trade_amount = min(available_balance * final_ratio, self.max_trade_amount)
        trade_amount = max(trade_amount, self.min_trade_amount)
        
        return min(trade_amount, available_balance)

    def should_buy(self, trend_score: float, volatility: float, current_time: datetime = None) -> bool:
        """買いシグナルの判定"""
        if len(self.trend_memory) < self.trend_memory_size:
            return False
            
        if current_time:
            if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.min_trade_interval:
                return False
            if current_time.hour in self.avoid_hours:
                return False

        trend_consistency = all(score < -0.4 for score in self.trend_memory[-3:])
        avg_trend = np.mean(self.trend_memory)
        recent_trend = np.mean(self.trend_memory[-2:])
        
        # 好ましい時間帯の条件
        if current_time and current_time.hour in self.preferred_hours:
            return (
                trend_score < -0.35 and
                avg_trend < -0.3 and
                recent_trend < -0.4 and
                volatility > 0.25 and
                volatility < 0.6 and
                self.current_rsi < self.rsi_oversold and
                trend_consistency
            )
        
        # 通常の条件
        return (
            trend_score < -0.5 and
            avg_trend < -0.45 and
            recent_trend < -0.5 and
            volatility > 0.3 and
            volatility < 0.5 and
            self.current_rsi < self.rsi_oversold and
            trend_consistency
        )

    def should_sell(self, trend_score: float, current_time: datetime = None) -> bool:
        """売りシグナルの判定"""
        if len(self.trend_memory) < self.trend_memory_size:
            return False
            
        if current_time:
            if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.min_trade_interval:
                return False
            if current_time.hour in self.avoid_hours:
                return False

        trend_consistency = all(score > 0.4 for score in self.trend_memory[-3:])
        avg_trend = np.mean(self.trend_memory)
        recent_trend = np.mean(self.trend_memory[-2:])
        
        # 好ましい時間帯の条件
        if current_time and current_time.hour in self.preferred_hours:
            return (
                trend_score > 0.35 and
                avg_trend > 0.3 and
                recent_trend > 0.4 and
                self.current_rsi > self.rsi_overbought and
                trend_consistency
            )
        
        # 通常の条件
        return (
            trend_score > 0.5 and
            avg_trend > 0.45 and
            recent_trend > 0.5 and
            self.current_rsi > self.rsi_overbought and
            trend_consistency
        )

    def calculate_position_size(self, price: float, amount: float) -> float:
        """ポジションサイズの計算（BTCの数量を5桁に切り上げ）"""
        btc_qty = amount / price
        return math.ceil(btc_qty * 100000) / 100000 