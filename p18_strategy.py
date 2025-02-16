"""
取引戦略のコアロジックを含むクラス
RSI、ボリンジャーバンドなどの技術指標の計算
トレンド分析
取引シグナルの判定
ポジションサイズの計算
損切り・利確の判断
"""
import numpy as np
from typing import List, Tuple
import math

class TradingStrategy:
    def __init__(self):
        self.rsi_period = 14     # RSIの期間
        self.rsi_oversold = 30   # RSI売られすぎの閾値
        self.rsi_overbought = 70 # RSI買われすぎの閾値
        self.stop_loss_pct = 0.5 # 損切りライン（0.5%）
        self.take_profit_pct = 1.0 # 利確ライン（1.0%）
        self.min_trade_amount = 5  # 最小取引額（USDT）
        self.max_trade_amount = 25  # 最大取引額を25 USDTに設定（総資産の25%まで）
        self.history_size = 60   # 5分間の価格履歴（5秒×60）

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
        """価格トレンドとボラティリティを計算"""
        if len(price_history) < self.history_size:
            return 0, 0, 0
        
        # RSIの計算
        rsi = self.calculate_rsi(price_history, self.rsi_period)
        
        # ボリンジャーバンドの計算
        upper, middle, lower = self.calculate_bollinger_bands(price_history)
        
        # 現在価格のボラティリティ位置を計算
        current_price = price_history[-1]
        band_width = upper - lower
        if band_width == 0:
            volatility = 0
        else:
            volatility = (current_price - lower) / band_width
        
        try:
            # 短期トレンド（1分）
            short_term_change = (price_history[-1] - price_history[-12]) / price_history[-12] * 100
            
            # 中期トレンド（5分）
            long_term_change = (price_history[-1] - price_history[0]) / price_history[0] * 100
            
            # トレンドスコアを計算（RSIとトレンドの組み合わせ）
            trend_score = (
                (rsi - 50) * 0.3 +  # RSIの影響（中立点からの乖離）
                short_term_change * 0.4 +  # 短期トレンドの影響
                long_term_change * 0.3  # 中期トレンドの影響
            )
            
            return trend_score, volatility, short_term_change
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
        """最適な取引量を計算"""
        if available_balance < self.min_trade_amount:
            return 0
        
        # トレンド強度による基本取引率の決定
        trend_strength = abs(trend_score)
        if trend_strength > 0.5:  # 強いトレンド
            base_ratio = 0.5  # 50%まで
        else:
            base_ratio = 0.3  # 30%まで
        
        # ボラティリティによる調整（0.7-1.0の範囲）
        volatility_factor = 0.7 + (0.3 * (1.0 - volatility))
        
        # 最終的な取引率を計算
        final_ratio = base_ratio * volatility_factor
        
        # 取引額を計算（最小取引額を考慮）
        trade_amount = min(available_balance * final_ratio, self.max_trade_amount)
        trade_amount = max(trade_amount, self.min_trade_amount)
        
        # 利用可能残高を超えないように調整
        trade_amount = min(trade_amount, available_balance)
        
        # 最小取引額以上になるように調整
        if trade_amount < self.min_trade_amount:
            if available_balance >= self.min_trade_amount:
                trade_amount = self.min_trade_amount
            else:
                trade_amount = 0
        
        return trade_amount

    def should_buy(self, trend_score: float, volatility: float) -> bool:
        """買いシグナルの判定"""
        return trend_score < -0.2 and volatility < 0.7

    def should_sell(self, trend_score: float) -> bool:
        """売りシグナルの判定"""
        return trend_score > 0.2

    def calculate_position_size(self, price: float, amount: float) -> float:
        """ポジションサイズの計算（BTCの数量を5桁に切り上げ）"""
        btc_qty = amount / price
        return math.ceil(btc_qty * 100000) / 100000 