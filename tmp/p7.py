#cursorのsonnetで生成した

import ccxt
import time
from datetime import datetime
import sys
import numpy as np

class TradingBot:
    def __init__(self, initial_balance):
        self.exchange = ccxt.binance()  # 価格取得用
        self.jpy_balance = initial_balance
        self.btc_balance = 0
        self.fee_rate = 0.001  # 0.1%の取引手数料
        self.price_history = []
        self.ma_short = 5  # 短期移動平均の期間
        self.ma_long = 20  # 長期移動平均の期間
        self.min_order = 0.00000001
        self.max_order = 20
        self.profit_threshold = 0.005  # 0.5%の利益を目標
        self.last_trade_price = None
        self.last_trade_type = None
        self.last_trade_time = None
        self.min_trade_interval = 5  # 最小取引間隔（秒）
        self.trade_amount_ratio = 0.3  # 取引量を30%に設定

    def get_btc_price(self):
        ticker = self.exchange.fetch_ticker('BTC/USDT')
        price = ticker['last'] * 150
        self.price_history.append(price)
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
        return price

    def calculate_total_assets(self, current_price):
        return self.jpy_balance + (self.btc_balance * current_price)

    def calculate_ma(self, period):
        if len(self.price_history) < period:
            return None
        return sum(self.price_history[-period:]) / period

    def calculate_rsi(self, period=14):
        if len(self.price_history) < period + 1:
            return None
        
        deltas = np.diff(self.price_history[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, period=20):
        if len(self.price_history) < period:
            return None, None, None
        
        prices = np.array(self.price_history[-period:])
        ma = np.mean(prices)
        std = np.std(prices)
        
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        
        return ma, upper, lower

    def can_trade(self):
        if self.last_trade_time is None:
            return True
        return time.time() - self.last_trade_time >= self.min_trade_interval

    def should_buy(self, current_price):
        if not self.can_trade() or len(self.price_history) < self.ma_long or self.jpy_balance <= 1000:
            return False

        rsi = self.calculate_rsi()
        if rsi is None or rsi > 30:  # RSIが30以下でないと買わない
            return False
        
        ma, upper, lower = self.calculate_bollinger_bands()
        if ma is None:
            return False

        ma_short = self.calculate_ma(self.ma_short)
        ma_long = self.calculate_ma(self.ma_long)
        if not ma_short or not ma_long:
            return False

        # トレンドと価格条件の確認
        trend_signal = ma_short > ma_long
        bb_signal = current_price < lower
        
        # 前回の取引との価格差を確認
        price_condition = True
        if self.last_trade_price and self.last_trade_type == "売り":
            price_condition = current_price < self.last_trade_price * (1 - self.profit_threshold)
        
        return (rsi < 30 and bb_signal) and trend_signal and price_condition

    def should_sell(self, current_price):
        if not self.can_trade() or len(self.price_history) < self.ma_long or self.btc_balance <= self.min_order:
            return False

        rsi = self.calculate_rsi()
        if rsi is None or rsi < 70:  # RSIが70以上でないと売らない
            return False
        
        ma, upper, lower = self.calculate_bollinger_bands()
        if ma is None:
            return False

        ma_short = self.calculate_ma(self.ma_short)
        ma_long = self.calculate_ma(self.ma_long)
        if not ma_short or not ma_long:
            return False

        # トレンドと価格条件の確認
        trend_signal = ma_short < ma_long
        bb_signal = current_price > upper
        
        # 前回の取引との価格差を確認
        price_condition = True
        if self.last_trade_price and self.last_trade_type == "買い":
            price_condition = current_price > self.last_trade_price * (1 + self.profit_threshold)
        
        return (rsi > 70 and bb_signal) and trend_signal and price_condition

    def execute_trade(self, current_price, is_buy):
        if is_buy:
            amount_in_jpy = self.jpy_balance * self.trade_amount_ratio
            btc_amount = amount_in_jpy / current_price
            
            if btc_amount < self.min_order:
                return False
            if btc_amount > self.max_order:
                btc_amount = self.max_order

            fee = btc_amount * self.fee_rate
            total_btc = btc_amount - fee
            total_jpy = amount_in_jpy

            if total_jpy <= self.jpy_balance:
                self.btc_balance += total_btc
                self.jpy_balance -= total_jpy
                self.last_trade_price = current_price
                self.last_trade_type = "買い"
                self.last_trade_time = time.time()
                self.print_trade_info(current_price, "買い")
                return True

        else:  # sell
            if self.btc_balance < self.min_order:
                return False
            
            amount_to_sell = self.btc_balance * self.trade_amount_ratio
            if amount_to_sell > self.max_order:
                amount_to_sell = self.max_order

            fee = amount_to_sell * self.fee_rate
            total_btc = amount_to_sell
            total_jpy = (amount_to_sell - fee) * current_price

            self.btc_balance -= total_btc
            self.jpy_balance += total_jpy
            self.last_trade_price = current_price
            self.last_trade_type = "売り"
            self.last_trade_time = time.time()
            self.print_trade_info(current_price, "売り")
            return True

        return False

    def print_trade_info(self, current_price, trade_type):
        print(f"\n取引実行: {trade_type}")
        print(f"現在のBTC価格：{current_price:,.0f}円")
        print(f"JPY残高: {self.jpy_balance:,.0f}円")
        print(f"BTC保有量：{self.btc_balance:.8f}BTC")
        total_assets = self.calculate_total_assets(current_price)
        print(f"総資産額：{total_assets:,.0f}円")
        
        if self.last_trade_price:
            profit_rate = ((total_assets - self.jpy_balance) / self.jpy_balance) * 100
            print(f"収益率: {profit_rate:+.2f}%")
        
        rsi = self.calculate_rsi()
        ma, upper, lower = self.calculate_bollinger_bands()
        if rsi is not None and ma is not None:
            print(f"RSI: {rsi:.1f}")
            print(f"ボリンジャーバンド - 上限: {upper:,.0f} 中央: {ma:,.0f} 下限: {lower:,.0f}")

    def run(self):
        print("取引シミュレーションを開始します...")
        print("初期設定:")
        print(f"- 利益目標: {self.profit_threshold*100:.1f}%")
        print(f"- 取引量: 利用可能額の{self.trade_amount_ratio*100:.0f}%")
        print(f"- RSI期間: 14")
        print(f"- ボリンジャーバンド期間: 20")
        print(f"- 最小取引間隔: {self.min_trade_interval}秒")
        
        while True:
            try:
                current_price = self.get_btc_price()
                
                if len(self.price_history) > 1:
                    price_change = ((current_price - self.price_history[-2]) / self.price_history[-2]) * 100
                    
                    if not self.execute_trade(current_price, self.should_buy(current_price)) and \
                       not self.execute_trade(current_price, self.should_sell(current_price)):
                        rsi = self.calculate_rsi()
                        ma, upper, lower = self.calculate_bollinger_bands()
                        market_info = ""
                        if rsi is not None and ma is not None:
                            bb_deviation = ((current_price - ma) / ma) * 100
                            market_info = f" | RSI: {rsi:.1f} | BB乖離: {bb_deviation:+.2f}%"
                        print(f"\r価格変動: {price_change:+.2f}%{market_info}", end="")
                
                time.sleep(1)

            except KeyboardInterrupt:
                print("\n\nプログラムを終了します...")
                final_assets = self.calculate_total_assets(current_price)
                initial_assets = self.jpy_balance
                profit_rate = ((final_assets - initial_assets) / initial_assets) * 100
                print(f"最終総資産: {final_assets:,.0f}円")
                print(f"総収益率: {profit_rate:+.2f}%")
                sys.exit()

            except Exception as e:
                print(f"エラーが発生しました: {e}")
                time.sleep(5)

if __name__ == "__main__":
    try:
        initial_balance = float(input("初期資金を入力してください（円）: "))
        bot = TradingBot(initial_balance)
        bot.run()
    except ValueError:
        print("有効な数値を入力してください。")
