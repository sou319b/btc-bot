#clineで生成した。
import ccxt
import time
import sys
from datetime import datetime

class TradingBot:
    def __init__(self, initial_balance=10000):
        self.exchange = ccxt.binance()  # 価格取得用
        self.jpy_balance = initial_balance  # 初期資金：1万円
        self.btc_balance = 0.0  # BTC保有量
        self.fee_rate = 0.0015  # 取引手数料0.15%
        self.min_order = 0.00000001  # 最小発注数量
        self.max_order = 20  # 最大発注数量
        self.last_price = None
        self.initial_balance = initial_balance
        self.price_history = []
        self.buy_threshold = -0.0005  # 0.05%下落で買い
        self.sell_threshold = 0.001  # 0.1%上昇で売り

    def get_btc_price(self):
        try:
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            # USDTからJPYへの概算換算（1USDT≒150円と仮定）
            return ticker['last'] * 150
        except Exception as e:
            print(f"価格取得エラー: {e}")
            return None

    def calculate_total_assets(self, current_price):
        return self.jpy_balance + (self.btc_balance * current_price)

    def calculate_fee(self, amount, price):
        return amount * price * self.fee_rate

    def execute_trade(self, action, price, amount):
        fee = self.calculate_fee(amount, price)
        
        if action == "BUY":
            total_cost = (amount * price) + fee
            if total_cost <= self.jpy_balance:
                self.jpy_balance -= total_cost
                self.btc_balance += amount
                return True
        else:  # SELL
            if amount <= self.btc_balance:
                total_revenue = (amount * price) - fee
                self.jpy_balance += total_revenue
                self.btc_balance -= amount
                return True
        return False

    def calculate_profit_loss(self, current_price):
        total_assets = self.calculate_total_assets(current_price)
        profit_loss = total_assets - self.initial_balance
        profit_loss_percentage = (profit_loss / self.initial_balance) * 100
        return profit_loss, profit_loss_percentage

    def print_status(self, current_price, action="", price_change=0):
        total_assets = self.calculate_total_assets(current_price)
        profit_loss, profit_loss_percentage = self.calculate_profit_loss(current_price)
        
        print(f"\n{'='*50}")
        print(f"現在のBTC価格：{current_price:,.0f}円")
        print(f"価格変動: {price_change*100:.3f}%")
        print(f"JPY残高: {self.jpy_balance:,.0f}円")
        print(f"BTC保有量：{self.btc_balance:.8f}BTC")
        print(f"総資産額：{total_assets:,.0f}円")
        print(f"損益：{profit_loss:,.0f}円 ({profit_loss_percentage:.2f}%)")
        if action:
            print(f"取引：{action}")
        print(f"{'='*50}\n")

    def run(self):
        print("取引ボットを開始します...")
        
        while True:
            try:
                current_price = self.get_btc_price()
                if current_price is None:
                    continue

                if self.last_price is None:
                    self.last_price = current_price
                    self.print_status(current_price)
                    continue

                # 価格変動を計算
                price_change = (current_price - self.last_price) / self.last_price

                # 買いシグナル: 価格が下落
                if price_change <= self.buy_threshold and self.jpy_balance > 0:
                        # 利用可能な資金の20%を使用
                        amount = min(
                            (self.jpy_balance * 0.2) / current_price,
                            self.max_order
                        )
                        amount = max(amount, self.min_order)
                        
                        if self.execute_trade("BUY", current_price, amount):
                            self.print_status(current_price, "買い", price_change)

                # 売りシグナル: 価格が上昇
                elif price_change >= self.sell_threshold and self.btc_balance > 0:
                        # 保有BTCの20%を売却
                        amount = min(self.btc_balance * 0.2, self.max_order)
                        amount = max(amount, self.min_order)
                        
                        if self.execute_trade("SELL", current_price, amount):
                            self.print_status(current_price, "売り", price_change)

                else:
                    self.print_status(current_price, price_change=price_change)

                self.last_price = current_price
                time.sleep(3)  # 3秒待機

            except KeyboardInterrupt:
                print("\n取引ボットを終了します...")
                break
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                time.sleep(1)

def parse_amount(amount_str):
    # 金額文字列から数値を抽出（例：'1万円' → 10000）
    amount_str = amount_str.replace(',', '')
    
    # 単位変換テーブル
    units = {
        '万': 10000,
        '億': 100000000
    }
    
    # 数値部分と単位部分を分離
    numeric = ''
    unit = 1
    for char in amount_str:
        if char.isdigit() or char == '.':
            numeric += char
        elif char in units:
            unit = units[char]
    
    if not numeric:
        return None
    
    try:
        return int(float(numeric) * unit)
    except ValueError:
        return None

if __name__ == "__main__":
    # デフォルトの初期資金
    initial_balance = 10000

    if len(sys.argv) > 1:
        amount = parse_amount(sys.argv[1])
        if amount is not None:
            initial_balance = amount
        else:
            print("無効な金額形式です。例：'1万円' または '10000'")
            print(f"デフォルト金額の{initial_balance:,}円を使用します。")
    
    print(f"初期資金: {initial_balance:,}円")
    bot = TradingBot(initial_balance=initial_balance)
    bot.run()
