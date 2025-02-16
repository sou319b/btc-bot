#claudeで生成した。
import ccxt
import time
import random
import numpy as np
from collections import deque
from datetime import datetime

# 初期設定
INITIAL_JPY = 1000000  # 初期資金100万円
BTC_HOLDINGS = 0
TICK_INTERVAL = 3  # 3秒ごとに取引
TRADING_FEE = 0.001  # 0.1%の取引手数料
MA_PERIOD = 20  # 移動平均の期間
VOLATILITY_PERIOD = 20  # ボラティリティ計算期間
BASE_THRESHOLD = 0.001  # 基本閾値（0.1%）
MAX_PURCHASE_AMOUNT = 50000  # 1回の最大購入額
SELL_PORTION = 0.5  # 売却時の保有量に対する割合

class TradingBot:
    def __init__(self):
        self.api = ccxt.bitflyer()
        self.price_history = deque(maxlen=max(MA_PERIOD, VOLATILITY_PERIOD))
        self.best_price = None
        self.jpy_balance = INITIAL_JPY
        self.btc_holdings = BTC_HOLDINGS

    def get_btc_price(self, retries=3, delay=5):
        """現在のBTC価格を取得"""
        for i in range(retries):
            try:
                ticker = self.api.fetch_ticker('BTC/JPY')
                return ticker['last']
            except Exception as e:
                if i == retries - 1:
                    raise
                print(f"価格取得失敗 ({i+1}/{retries}): {str(e)}")
                time.sleep(delay)

    def calculate_moving_average(self):
        """移動平均を計算"""
        if len(self.price_history) < MA_PERIOD:
            return None
        return np.mean(list(self.price_history)[-MA_PERIOD:])

    def calculate_volatility(self):
        """ボラティリティを計算"""
        if len(self.price_history) < VOLATILITY_PERIOD:
            return None
        prices = list(self.price_history)[-VOLATILITY_PERIOD:]
        return np.std(prices) / np.mean(prices)

    def get_dynamic_threshold(self):
        """ボラティリティに基づく動的閾値を計算"""
        volatility = self.calculate_volatility()
        if volatility is None:
            return BASE_THRESHOLD
        return BASE_THRESHOLD * (1 + volatility)

    def should_buy(self, current_price, ma_price):
        """買いシグナルの判定"""
        if not ma_price:
            return False
        
        threshold = self.get_dynamic_threshold()
        price_diff = (current_price - self.best_price) / self.best_price
        trend_diff = (current_price - ma_price) / ma_price
        
        return (price_diff < -threshold and  # 価格が閾値以上下落
                trend_diff < 0 and  # 移動平均を下回っている
                self.jpy_balance > 0)  # 購入資金がある

    def should_sell(self, current_price, ma_price):
        """売りシグナルの判定"""
        if not ma_price:
            return False
            
        threshold = self.get_dynamic_threshold()
        price_diff = (current_price - self.best_price) / self.best_price
        trend_diff = (current_price - ma_price) / ma_price
        
        return (price_diff > threshold and  # 価格が閾値以上上昇
                trend_diff > 0 and  # 移動平均を上回っている
                self.btc_holdings > 0)  # 売却可能なBTCがある

    def print_status(self, price, action, extra_info=""):
        """現在の状態を表示し、ログに保存"""
        total_assets = self.jpy_balance + (self.btc_holdings * price)
        log_entry = f"""
[取引時刻] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
現在のBTC価格：{price:,.0f} JPY
JPY残高: {self.jpy_balance:,.0f} JPY
BTC保有量：{self.btc_holdings:.8f} BTC
総資産額：{total_assets:,.0f} JPY
アクション：{action}
{extra_info}
-------------------------
"""
        print(log_entry.strip())
        
        try:
            with open('trading_log.txt', 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"ログファイル書き込みエラー: {e}")

    def execute_trade(self):
        """取引の実行"""
        try:
            base_price = self.get_btc_price()
            # ランダムな価格変動を追加 (±0.2%)
            current_price = base_price * (1 + (random.random() * 0.004 - 0.002))
            
            # 価格履歴を更新
            self.price_history.append(current_price)
            if self.best_price is None:
                self.best_price = current_price
                
            # 価格履歴が十分でない場合は取引を見送る
            if len(self.price_history) < max(MA_PERIOD, VOLATILITY_PERIOD):
                print(f"価格履歴を蓄積中... ({len(self.price_history)}/{max(MA_PERIOD, VOLATILITY_PERIOD)})")
                return True
                
            ma_price = self.calculate_moving_average()
            threshold = self.get_dynamic_threshold()
            
            # 移動平均価格のフォーマット
            ma_price_str = f"{ma_price:,.0f}" if ma_price is not None else "計算中"
            extra_info = f"""
移動平均価格: {ma_price_str} JPY
現在の閾値: {threshold*100:.3f}%
"""

            # 買い判定
            if self.should_buy(current_price, ma_price):
                buy_amount = min(self.jpy_balance, MAX_PURCHASE_AMOUNT)
                actual_buy_amount = buy_amount * (1 - TRADING_FEE)  # 手数料考慮
                btc_bought = actual_buy_amount / current_price
                self.jpy_balance -= buy_amount
                self.btc_holdings += btc_bought
                self.best_price = current_price
                self.print_status(current_price, f"買い {btc_bought:.8f} BTC", extra_info)

            # 売り判定
            elif self.should_sell(current_price, ma_price):
                sell_amount = self.btc_holdings * SELL_PORTION
                received_amount = sell_amount * current_price * (1 - TRADING_FEE)  # 手数料考慮
                self.jpy_balance += received_amount
                self.btc_holdings -= sell_amount
                self.best_price = current_price
                self.print_status(current_price, f"売り {sell_amount:.8f} BTC", extra_info)

        except Exception as e:
            print(f"取引実行エラー: {e}")
            return False
        return True

    def run(self):
        """トレーディングボットの実行"""
        print("トレーディングボット開始...")
        
        # ログファイル初期化
        try:
            with open('trading_log.txt', 'w', encoding='utf-8') as f:
                f.write(f"シミュレーション開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"初期資金: {INITIAL_JPY:,} JPY\n\n")
        except Exception as e:
            print(f"ログファイル初期化エラー: {e}")
            return

        # API接続テスト
        try:
            print("API接続テスト中...")
            test_price = self.get_btc_price()
            print(f"初期価格取得成功: {test_price:,} JPY")
        except Exception as e:
            print(f"API接続エラー: {str(e)}")
            return

        while True:
            try:
                if not self.execute_trade():
                    time.sleep(60)  # エラー時は1分待機
                    continue
                    
                time.sleep(TICK_INTERVAL)
                
            except KeyboardInterrupt:
                print("\nシミュレーションを終了します")
                final_price = self.get_btc_price()
                final_total = self.jpy_balance + (self.btc_holdings * final_price)
                profit = final_total - INITIAL_JPY
                print(f"最終資産額: {final_total:,} JPY")
                print(f"利益: {profit:,} JPY")
                
                try:
                    with open('trading_log.txt', 'a', encoding='utf-8') as f:
                        f.write(f"\nシミュレーション終了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"最終資産額: {final_total:,} JPY\n")
                        f.write(f"利益: {profit:,} JPY\n")
                except Exception as e:
                    print(f"最終結果のログ記録エラー: {e}")
                break

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()