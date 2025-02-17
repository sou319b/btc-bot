"""
取引戦略のテンプレート

このファイルは、取引戦略を実装するための基本構造を示しています。
新しい戦略を作成する場合は、このテンプレートをベースにしてください。

基本的な使い方：
1. このテンプレートをコピーして新しいファイルを作成（例：p18_strategy_v2.py）
2. TradingStrategyクラスを継承して新しい戦略クラスを作成
3. 必要なメソッドをオーバーライドして独自の戦略を実装
"""

import numpy as np
from typing import List, Tuple
import math

class TradingStrategyBase:
    """取引戦略の基本クラス"""
    
    def __init__(self):
        # 基本パラメータ（必要に応じて調整）
        self.min_trade_amount = 5    # 最小取引額（USDT）
        self.max_trade_amount = 25   # 最大取引額（USDT）
        self.history_size = 60       # 価格履歴のサイズ
        
        # 損切り・利確の設定
        self.stop_loss_pct = 0.5     # 損切りライン（%）
        self.take_profit_pct = 1.0   # 利確ライン（%）
        
        # 独自のパラメータを追加
        self.setup_parameters()
    
    def setup_parameters(self):
        """
        戦略固有のパラメータを設定
        このメソッドをオーバーライドして独自のパラメータを追加
        """
        pass
    
    def calculate_indicators(self, price_history: List[float]) -> dict:
        """
        テクニカル指標を計算
        このメソッドをオーバーライドして独自の指標を計算

        Args:
            price_history: 価格履歴のリスト

        Returns:
            dict: 計算された指標の辞書
        """
        return {}
    
    def should_buy(self, indicators: dict) -> bool:
        """
        買いシグナルの判定
        このメソッドをオーバーライドして独自の買い条件を実装

        Args:
            indicators: calculate_indicatorsで計算された指標

        Returns:
            bool: 買いシグナルがTrue、そうでない場合はFalse
        """
        return False
    
    def should_sell(self, indicators: dict) -> bool:
        """
        売りシグナルの判定
        このメソッドをオーバーライドして独自の売り条件を実装

        Args:
            indicators: calculate_indicatorsで計算された指標

        Returns:
            bool: 売りシグナルがTrue、そうでない場合はFalse
        """
        return False
    
    def calculate_position_size(self, price: float, amount: float) -> float:
        """
        ポジションサイズの計算
        必要に応じてオーバーライドして独自の計算方法を実装

        Args:
            price: 現在の価格
            amount: 取引金額（USDT）

        Returns:
            float: BTCの数量（5桁に丸める）
        """
        btc_qty = amount / price
        return math.ceil(btc_qty * 100000) / 100000
    
    def should_close_position(self, current_price: float, entry_price: float) -> bool:
        """
        ポジションクローズの判断
        必要に応じてオーバーライドして独自の判断ロジックを実装

        Args:
            current_price: 現在の価格
            entry_price: エントリー価格

        Returns:
            bool: クローズすべき場合はTrue、そうでない場合はFalse
        """
        if entry_price is None:
            return False
        
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        return (price_change_pct < -self.stop_loss_pct or  # 損切り
                price_change_pct > self.take_profit_pct)    # 利確
    
    def calculate_optimal_trade_amount(self, current_price: float, 
                                     indicators: dict, 
                                     available_balance: float) -> float:
        """
        最適な取引量を計算
        必要に応じてオーバーライドして独自の計算方法を実装

        Args:
            current_price: 現在の価格
            indicators: 計算された指標
            available_balance: 利用可能な残高

        Returns:
            float: 取引金額（USDT）
        """
        if available_balance < self.min_trade_amount:
            return 0
        
        # デフォルトでは利用可能残高の30%を使用
        trade_amount = available_balance * 0.3
        
        # 最小・最大取引額の制限を適用
        trade_amount = min(trade_amount, self.max_trade_amount)
        trade_amount = max(trade_amount, self.min_trade_amount)
        
        return min(trade_amount, available_balance)


# 実際の戦略の実装例
class RSIStrategy(TradingStrategyBase):
    """RSIを使用した取引戦略の例"""
    
    def setup_parameters(self):
        """RSI戦略のパラメータを設定"""
        self.rsi_period = 14        # RSIの期間
        self.rsi_oversold = 30      # 売られすぎの閾値
        self.rsi_overbought = 70    # 買われすぎの閾値
    
    def calculate_rsi(self, prices: List[float]) -> float:
        """RSIを計算"""
        if len(prices) < self.rsi_period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_indicators(self, price_history: List[float]) -> dict:
        """指標を計算"""
        if len(price_history) < self.history_size:
            return {'rsi': 50.0, 'trend': 0}
        
        rsi = self.calculate_rsi(price_history)
        
        # トレンドの計算（単純な例）
        short_term_change = (price_history[-1] - price_history[-5]) / price_history[-5] * 100
        
        return {
            'rsi': rsi,
            'trend': short_term_change
        }
    
    def should_buy(self, indicators: dict) -> bool:
        """買いシグナルの判定"""
        rsi = indicators.get('rsi', 50)
        trend = indicators.get('trend', 0)
        
        # RSIが売られすぎで、トレンドが上昇傾向
        return rsi < self.rsi_oversold and trend > 0
    
    def should_sell(self, indicators: dict) -> bool:
        """売りシグナルの判定"""
        rsi = indicators.get('rsi', 50)
        trend = indicators.get('trend', 0)
        
        # RSIが買われすぎで、トレンドが下降傾向
        return rsi > self.rsi_overbought and trend < 0


# 新しい戦略を作成する際のテンプレート
class NewStrategy(TradingStrategyBase):
    """新しい取引戦略のテンプレート"""
    
    def setup_parameters(self):
        """
        戦略のパラメータを設定
        必要なパラメータをここで定義
        """
        # 例：
        self.ma_short = 10    # 短期移動平均の期間
        self.ma_long = 30     # 長期移動平均の期間
    
    def calculate_indicators(self, price_history: List[float]) -> dict:
        """
        独自の指標を計算
        price_historyを使用して必要な指標を計算
        """
        # 例：
        if len(price_history) < self.ma_long:
            return {}
            
        ma_short = np.mean(price_history[-self.ma_short:])
        ma_long = np.mean(price_history[-self.ma_long:])
        
        return {
            'ma_short': ma_short,
            'ma_long': ma_long,
            'price': price_history[-1]
        }
    
    def should_buy(self, indicators: dict) -> bool:
        """
        買いシグナルの条件を定義
        """
        # 例：ゴールデンクロス
        if not all(k in indicators for k in ['ma_short', 'ma_long', 'price']):
            return False
            
        return (indicators['ma_short'] > indicators['ma_long'] and
                indicators['price'] > indicators['ma_short'])
    
    def should_sell(self, indicators: dict) -> bool:
        """
        売りシグナルの条件を定義
        """
        # 例：デッドクロス
        if not all(k in indicators for k in ['ma_short', 'ma_long', 'price']):
            return False
            
        return (indicators['ma_short'] < indicators['ma_long'] and
                indicators['price'] < indicators['ma_short']) 