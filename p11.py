import ccxt
import time

class TradingBot:
    def __init__(self, initial_jpy):
        self.exchange = ccxt.bitflyer()  # 取引所をbitflyerに変更
        self.jpy_balance = initial_jpy
        self.btc_balance = 0.0
        self.previous_price = None
        self.total_assets_history = []

    def fetch_btc_price(self):
        try:
            # オーダーブックから最新価格を取得
            order_book = self.exchange.fetch_order_book('BTC/JPY', limit=5)
            best_bid = order_book['bids'][0][0] if len(order_book['bids']) > 0 else None
            best_ask = order_book['asks'][0][0] if len(order_book['asks']) > 0 else None
            
            if best_bid and best_ask:
                current_price = (best_bid + best_ask) / 2
                print(f"デバッグ: 最新買い価格 - {best_bid} | 最新売り価格 - {best_ask} | 計算価格 - {current_price}")
                return current_price
            else:
                return self.previous_price or 0  # 前回価格を保持
        
        except Exception as e:
            print(f"オーダーブック取得エラー: {str(e)}")
            return self.previous_price or 0  # エラー時は前回価格を使用

    def calculate_price_change(self, current_price):
        if self.previous_price is None:
            self.previous_price = current_price  # 初回は現在価格で初期化
            return 0.0
        change = ((current_price - self.previous_price) / self.previous_price) * 100
        self.previous_price = current_price  # 変更: 価格更新をここで実行
        return change

    def execute_trade(self, action, price, amount):
        if action == 'buy':
            self.jpy_balance -= price * amount
            self.btc_balance += amount
        elif action == 'sell':
            self.jpy_balance += price * amount
            self.btc_balance -= amount

    def run(self):
        while True:
            try:
                current_price = self.fetch_btc_price()
                price_change = self.calculate_price_change(current_price)

                total_assets = self.jpy_balance + (self.btc_balance * current_price)
                self.total_assets_history.append(total_assets)

                # 取引判断ロジック（改良版）
                trade_amount = 0.0
                action = ''
                # 閾値を0.01%に変更（より敏感に反応）
                if price_change > 0.01:  # 微小な上昇でも買い
                    action = 'buy'
                    trade_amount = self.jpy_balance / current_price
                elif price_change < -0.01:  # 微小な下落でも売り
                    action = 'sell'
                    trade_amount = self.btc_balance

                if trade_amount > 0:
                    self.execute_trade(action, current_price, trade_amount)
                    print(f"\n現在のBTC価格：{current_price:,.0f} JPY")
                    print(f"JPY残高: {self.jpy_balance:,.0f} JPY")
                    print(f"BTC保有量：{self.btc_balance:.6f}")
                    print(f"総資産額：{total_assets:,.0f} JPY")
                    print(f"買/売：{action}")
                else:
                    print(f"価格変動: {price_change:+.2f}%", end='\r')

                time.sleep(3)  # 3秒間隔に変更

            except KeyboardInterrupt:
                print("\nシミュレーションを終了します")
                break
            except Exception as e:
                print(f"エラーが発生しました: {str(e)}")
                time.sleep(10)

if __name__ == "__main__":
    initial_funds = float(input("初期資金をJPYで入力してください: "))
    bot = TradingBot(initial_funds)
    bot.run()
