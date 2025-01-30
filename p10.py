import time
import ccxt

class BitcoinTradingBot:
    def __init__(self, initial_funds, fee_rate=0.001):
        self.initial_funds = initial_funds  # 初期資金（日本円）
        self.funds_jpy = initial_funds     # 現在の日本円残高
        self.btc_balance = 0.0             # 現在のBTC保有量
        self.fee_rate = fee_rate           # 手数料率 (0.1%)
        self.min_order_size = 0.00000001   # 最小発注数量
        self.max_order_size = 20           # 最大発注数量
        self.last_price = None             # 前回の価格を記録
        self.exchange = ccxt.bitflyer()    # 取引所API (ここではbitFlyerを使用)
        self.prices = []                   # 過去の価格を記録

    def fetch_btc_price(self):
        """BTCの現在価格を取得"""
        try:
            ticker = self.exchange.fetch_ticker('BTC/JPY')
            return ticker['last']
        except Exception as e:
            print(f"価格取得エラー: {e}")
            return None

    def calculate_fee(self, amount):
        """手数料を計算"""
        return amount * self.fee_rate

    def trade(self, action, price, amount):
        """取引を実行"""
        if action == "buy":
            cost = amount * price
            fee = self.calculate_fee(cost)
            if self.funds_jpy >= cost + fee:
                self.funds_jpy -= cost + fee
                self.btc_balance += amount
                return True
        elif action == "sell":
            revenue = amount * price
            fee = self.calculate_fee(revenue)
            if self.btc_balance >= amount:
                self.btc_balance -= amount
                self.funds_jpy += revenue - fee
                return True
        return False

    def calculate_moving_average(self, prices, window):
        """移動平均を計算"""
        return sum(prices[-window:]) / window

    def simulate(self):
        """取引シミュレーション"""
        while True:
            current_price = self.fetch_btc_price()
            if current_price is None:
                print("価格取得エラー")
                time.sleep(5)
                continue

            self.prices.append(current_price)
            if len(self.prices) > 10:  # 過去10個の価格を使用
                short_ma = self.calculate_moving_average(self.prices, 5)  # 短期移動平均
                long_ma = self.calculate_moving_average(self.prices, 10)  # 長期移動平均

                if short_ma > long_ma:  # 短期が長期を上回ったら買い
                    action = "buy"
                elif short_ma < long_ma:  # 短期が長期を下回ったら売り
                    action = "sell"
                else:
                    action = None

                if action:
                    order_size = min(self.max_order_size, max(self.min_order_size, self.funds_jpy / current_price))
                    success = self.trade(action, current_price, order_size)
                    if success:
                        print(f"現在のBTC価格：{current_price}")
                        print(f"JPY残高: {self.funds_jpy:.2f}")
                        print(f"BTC保有量：{self.btc_balance:.8f}")
                        print(f"総資産額：{self.funds_jpy + self.btc_balance * current_price:.2f}")
                        print(f"買/売：{action}")
                        print(f"取引所の売買手数料：{self.calculate_fee(order_size * current_price):.8f} BTC")
                    else:
                        print("取引失敗: 資金またはBTCが不足しています。")

            time.sleep(1)  # 高頻度取引のため1秒ごとに更新

if __name__ == "__main__":
    # 初期資金を入力
    initial_funds = float(input("初期資金を入力してください（例: 100000）: "))
    bot = BitcoinTradingBot(initial_funds)
    bot.simulate()