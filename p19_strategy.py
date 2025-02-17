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
        self.rsi_oversold = 28
        self.rsi_overbought = 72
        self.stop_loss_pct = 0.2    # 損切りをさらに早めに
        self.take_profit_pct = 0.5   # 利確ラインを調整
        self.min_trade_amount = 15   # 最小取引額を調整
        self.max_trade_amount = 40   # 最大取引額を調整
        self.history_size = 90
        self.ma_short = 7
        self.ma_mid = 21
        self.ma_long = 50
        self.current_rsi = 50.0
        self.bb_period = 20
        self.bb_std = 2.0
        self.trend_memory = []
        self.trend_memory_size = 4
        self.last_trade_price = None
        self.last_trade_time = None
        self.min_trade_interval = 7200
        
        # 収益の高い時間帯に再最適化
        self.preferred_hours = {13, 4, 14}  # 新しい収益の高い時間帯
        self.avoid_hours = {11, 6, 16}    # 新しい損失の大きい時間帯

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
        """価格トレンドとボラティリティを計算（さらに最適化）"""
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
            # 価格変化率の計算（より多様な期間を考慮）
            price_changes = []
            for period in [3, 5, 10, 20, 30]:  # より多くの期間を追加
                if len(price_history) > period:
                    change = (price_history[-1] - price_history[-period]) / price_history[-period] * 100
                    price_changes.append(change)
                else:
                    price_changes.append(0)
            
            # 移動平均のアライメント分析（より厳格に）
            ma_alignment = 0
            if ma_short > ma_mid > ma_long and current_price > ma_short * 1.001:  # より厳格な条件
                ma_alignment = 2
            elif ma_short < ma_mid < ma_long and current_price < ma_short * 0.999:
                ma_alignment = -2
            
            # ボリンジャーバンドの分析（より厳格に）
            bb_score = 0
            if current_price <= lower * 0.998:  # より強い買いシグナル
                bb_score = -2
            elif current_price >= upper * 1.002:  # より強い売りシグナル
                bb_score = 2
            else:
                bb_score = (current_price - middle) / (upper - middle) if upper != middle else 0
            
            # トレンドスコアの計算（重みを調整）
            trend_score = (
                ((self.current_rsi - 50) / 22) * 0.15 +  # RSIの感度を調整
                np.mean(price_changes) * 0.40 +         # 価格変化の重みをさらに増加
                ma_alignment * 0.30 +                   # 移動平均の重みを調整
                bb_score * 0.15
            )
            
            # 時間帯による調整
            if current_time and current_time.hour in self.preferred_hours:
                trend_score *= 1.4  # 好ましい時間帯での信号をより強調
            
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
        """最適な取引量を計算（100USDT向けに最適化）"""
        if available_balance < self.min_trade_amount:
            return 0
        
        # 基本取引率の決定（より保守的に）
        trend_strength = abs(trend_score)
        if trend_strength > 0.6:
            base_ratio = 0.35  # より保守的な比率
        elif trend_strength > 0.4:
            base_ratio = 0.25
        else:
            base_ratio = 0.15
        
        # 好ましい時間帯では取引サイズを増加
        if current_time and current_time.hour in self.preferred_hours:
            base_ratio *= 1.2
        
        # トレンドの一貫性による調整
        if len(self.trend_memory) >= self.trend_memory_size:
            trend_std = np.std(self.trend_memory)
            consistency_factor = 1.0 - min(trend_std, 0.4)
        else:
            consistency_factor = 0.5
        
        # ボラティリティによる調整（より保守的に）
        volatility_factor = 0.8 + (0.2 * (1.0 - min(volatility, 0.4)))
        
        # RSIの極値による調整
        rsi_factor = 1.0
        if (self.current_rsi < 25 and trend_score < 0) or (self.current_rsi > 75 and trend_score > 0):
            rsi_factor = 1.15  # より控えめな増加
        
        final_ratio = base_ratio * consistency_factor * volatility_factor * rsi_factor
        
        trade_amount = min(available_balance * final_ratio, self.max_trade_amount)
        trade_amount = max(trade_amount, self.min_trade_amount)
        
        return min(trade_amount, available_balance)

    def should_buy(self, trend_score: float, volatility: float, current_time: datetime = None) -> bool:
        """買いシグナルの判定（より保守的に）"""
        if len(self.trend_memory) < self.trend_memory_size:
            return False
            
        if current_time:
            if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.min_trade_interval:
                return False
            if not self.is_good_trading_time(current_time):
                return False

        trend_consistency = all(score < -0.35 for score in self.trend_memory)  # より厳格な一貫性
        avg_trend = np.mean(self.trend_memory)
        
        # 好ましい時間帯の条件
        if current_time and current_time.hour in self.preferred_hours:
            return (
                trend_score < -0.4 and          # より厳格なスコア
                avg_trend < -0.35 and           # より厳格な平均トレンド
                volatility > 0.2 and
                volatility < 0.4 and
                self.current_rsi < self.rsi_oversold and
                trend_consistency
            )
        
        # 通常の条件
        return (
            trend_score < -0.65 and             # より厳格なスコア
            avg_trend < -0.55 and               # より厳格な平均トレンド
            volatility > 0.2 and
            volatility < 0.35 and
            self.current_rsi < self.rsi_oversold and
            trend_consistency
        )

    def should_sell(self, trend_score: float, current_time: datetime = None) -> bool:
        """売りシグナルの判定（より保守的に）"""
        if len(self.trend_memory) < self.trend_memory_size:
            return False
            
        if current_time:
            if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.min_trade_interval:
                return False
            if not self.is_good_trading_time(current_time):
                return False

        trend_consistency = all(score > 0.35 for score in self.trend_memory)  # より厳格な一貫性
        avg_trend = np.mean(self.trend_memory)
        
        # 好ましい時間帯の条件
        if current_time and current_time.hour in self.preferred_hours:
            return (
                trend_score > 0.4 and           # より厳格なスコア
                avg_trend > 0.35 and            # より厳格な平均トレンド
                self.current_rsi > self.rsi_overbought and
                trend_consistency
            )
        
        # 通常の条件
        return (
            trend_score > 0.65 and              # より厳格なスコア
            avg_trend > 0.55 and                # より厳格な平均トレンド
            self.current_rsi > self.rsi_overbought and
            trend_consistency
        )

    def calculate_position_size(self, price: float, amount: float) -> float:
        """ポジションサイズの計算（BTCの数量を5桁に切り上げ）"""
        btc_qty = amount / price
        return math.ceil(btc_qty * 100000) / 100000 