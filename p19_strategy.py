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

class TradingStrategy:
    def __init__(self):
        self.rsi_period = 14
        self.rsi_oversold = 35      # RSIの閾値を調整
        self.rsi_overbought = 65
        self.stop_loss_pct = 0.3    # 損切りラインを厳格化
        self.take_profit_pct = 0.6   # 利確ラインを調整
        self.min_trade_amount = 15   # 最小取引額を調整
        self.max_trade_amount = 100  # 最大取引額を増加
        self.history_size = 60       # より短期の価格履歴
        self.ma_short = 5           # 超短期移動平均
        self.ma_mid = 15            # 短期移動平均
        self.ma_long = 30           # 中期移動平均
        self.current_rsi = 50.0
        self.bb_period = 20
        self.bb_std = 2.0
        self.trend_memory = []      # トレンドの履歴
        self.trend_memory_size = 3   # トレンドメモリのサイズ
        self.last_trade_price = None  # 最後の取引価格を記録

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

    def calculate_trend(self, price_history: List[float]) -> Tuple[float, float, float]:
        """価格トレンドとボラティリティを計算（最適化バージョン）"""
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
            # 価格変化率の計算
            price_changes = []
            for period in [3, 5, 10]:  # より短期に焦点
                if len(price_history) > period:
                    change = (price_history[-1] - price_history[-period]) / price_history[-period] * 100
                    price_changes.append(change)
                else:
                    price_changes.append(0)
            
            # 移動平均のアライメント分析
            ma_alignment = 0
            if ma_short > ma_mid > ma_long:  # 上昇トレンド
                ma_alignment = 1
                if current_price > ma_short:  # 価格が全ての移動平均線を上回る
                    ma_alignment = 2
            elif ma_short < ma_mid < ma_long:  # 下降トレンド
                ma_alignment = -1
                if current_price < ma_short:  # 価格が全ての移動平均線を下回る
                    ma_alignment = -2
            
            # ボリンジャーバンドの分析
            bb_score = 0
            if current_price <= lower:  # 下限ブレイク
                bb_score = -1
            elif current_price >= upper:  # 上限ブレイク
                bb_score = 1
            else:
                bb_score = (current_price - middle) / (upper - middle) if upper != middle else 0
            
            # トレンドスコアの計算
            trend_score = (
                ((self.current_rsi - 50) / 25) * 0.2 +     # RSIの影響（正規化）
                np.mean(price_changes) * 0.3 +             # 短期価格変化
                ma_alignment * 0.3 +                       # 移動平均のアライメント
                bb_score * 0.2                            # ボリンジャーバンドの位置
            )
            
            # トレンドメモリの更新
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

    def calculate_optimal_trade_amount(self, current_price: float, trend_score: float, volatility: float, available_balance: float) -> float:
        """最適な取引量を計算（最適化バージョン）"""
        if available_balance < self.min_trade_amount:
            return 0
        
        # トレンド強度による基本取引率の決定
        trend_strength = abs(trend_score)
        if trend_strength > 0.5:     # 強いトレンド
            base_ratio = 0.5
        elif trend_strength > 0.3:    # 中程度のトレンド
            base_ratio = 0.3
        else:
            base_ratio = 0.2         # 弱いトレンド
        
        # トレンドの一貫性による調整
        if len(self.trend_memory) >= self.trend_memory_size:
            trend_std = np.std(self.trend_memory)
            consistency_factor = 1.0 - min(trend_std, 0.5)
        else:
            consistency_factor = 0.5
        
        # ボラティリティによる調整（0.7-1.0の範囲）
        volatility_factor = 0.7 + (0.3 * (1.0 - min(volatility, 0.5)))
        
        # RSIの極値による調整
        rsi_factor = 1.0
        if (self.current_rsi < 30 and trend_score < 0) or (self.current_rsi > 70 and trend_score > 0):
            rsi_factor = 1.2
        
        # 最終的な取引率を計算
        final_ratio = base_ratio * consistency_factor * volatility_factor * rsi_factor
        
        # 取引額を計算
        trade_amount = min(available_balance * final_ratio, self.max_trade_amount)
        trade_amount = max(trade_amount, self.min_trade_amount)
        
        return min(trade_amount, available_balance)

    def should_buy(self, trend_score: float, volatility: float) -> bool:
        """買いシグナルの判定（最適化バージョン）"""
        if len(self.trend_memory) < self.trend_memory_size:
            return False
        
        # トレンドの一貫性をチェック
        trend_consistency = all(score < -0.1 for score in self.trend_memory)
        avg_trend = np.mean(self.trend_memory)
        
        # 価格が前回の取引価格より低い場合のみ取引
        price_condition = True
        if self.last_trade_price is not None:
            price_condition = self.last_trade_price > price_history[-1]
        
        return (
            trend_score < -0.3 and           # 下降トレンド
            avg_trend < -0.2 and            # 平均トレンドも下降
            volatility > 0.1 and            # 最小ボラティリティ
            volatility < 0.5 and            # 最大ボラティリティ
            self.current_rsi < self.rsi_oversold and  # 売られすぎ
            trend_consistency and            # トレンドの一貫性
            price_condition                  # 価格条件
        )

    def should_sell(self, trend_score: float) -> bool:
        """売りシグナルの判定（最適化バージョン）"""
        if len(self.trend_memory) < self.trend_memory_size:
            return False
        
        # トレンドの一貫性をチェック
        trend_consistency = all(score > 0.1 for score in self.trend_memory)
        avg_trend = np.mean(self.trend_memory)
        
        # 価格が前回の取引価格より高い場合のみ取引
        price_condition = True
        if self.last_trade_price is not None:
            price_condition = self.last_trade_price < price_history[-1]
        
        return (
            trend_score > 0.3 and            # 上昇トレンド
            avg_trend > 0.2 and             # 平均トレンドも上昇
            self.current_rsi > self.rsi_overbought and  # 買われすぎ
            trend_consistency and            # トレンドの一貫性
            price_condition                  # 価格条件
        )

    def calculate_position_size(self, price: float, amount: float) -> float:
        """ポジションサイズの計算（BTCの数量を5桁に切り上げ）"""
        btc_qty = amount / price
        return math.ceil(btc_qty * 100000) / 100000 