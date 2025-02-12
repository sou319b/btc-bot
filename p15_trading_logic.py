import time
from datetime import datetime
import logging
from p14_bybit_handler import BybitHandler

class TradingBot:
    def __init__(self):
        self.bybit = BybitHandler()
        self.initial_total = None
        self.best_price = None
        self.setup_logging()

    def setup_logging(self):
        """ロギングの設定を行う"""
        log_filename = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def print_trade_info(self, action, current_time, current_price, usdt_balance, btc_holding):
        total_assets = usdt_balance + btc_holding * current_price
        
        log_message = f"\n━━━━━━━━━━ 取引情報 ━━━━━━━━━━\n"
        log_message += f"📅 時刻　　　：{current_time}\n"
        log_message += f"💰 BTC価格　：{current_price:,.2f} USDT\n"
        log_message += f"💵 USDT残高 ：{usdt_balance:,.2f} USDT\n"
        log_message += f"₿ BTC保有量：{btc_holding:.6f} BTC\n"
        log_message += f"📊 総資産額 ：{total_assets:,.2f} USDT\n"
        
        if self.initial_total is not None:
            profit = total_assets - self.initial_total
            profit_percentage = (profit / self.initial_total) * 100
            if profit >= 0:
                log_message += f"💹 現在の利益：+{profit:,.2f} USDT (+{profit_percentage:.2f}%)\n"
            else:
                log_message += f"📉 現在の損失：{profit:,.2f} USDT ({profit_percentage:.2f}%)\n"
        
        if action != "情報":
            log_message += f"📈 取引種別 ：{action}\n"
        log_message += "━━━━━━━━━━━━━━━━━━━━━━━"
        
        print(log_message)
        self.logger.info(log_message)

    def initialize(self):
        self.logger.info("初期価格を取得中...")
        print("初期価格を取得中...")
        initial_price = None
        while initial_price is None:
            initial_price = self.bybit.get_btc_price()
            if initial_price is None:
                time.sleep(1)
        
        # 初期資産を計算
        initial_usdt, initial_btc = self.bybit.get_wallet_info()
        self.initial_total = initial_usdt + initial_btc * initial_price
        self.best_price = initial_price
        
        print(f"\n━━━━━━━━━━ 初期状態 ━━━━━━━━━━")
        print(f"💫 初期BTC価格：{initial_price:,.2f} USDT")
        print(f"💰 初期総資産　：{self.initial_total:,.2f} USDT")
        print("━━━━━━━━━━━━━━━━━━━━━━━\n")

    def execute_trade(self):
        try:
            while True:
                current_price = self.bybit.get_btc_price()
                if current_price is None:
                    error_msg = "価格の取得に失敗しました。再試行します..."
                    print(error_msg)
                    self.logger.error(error_msg)
                    time.sleep(1)
                    continue

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                price_diff = ((current_price - self.best_price) / self.best_price) * 100
                
                price_info = f"\n━━━━━━━━━━ 価格情報 ━━━━━━━━━━\n"
                price_info += f"📅 時刻　　　：{current_time}\n"
                price_info += f"💰 現在価格　：{current_price:,.2f} USDT\n"
                if price_diff > 0:
                    price_info += f"📈 価格変動　：+{price_diff:.2f}%\n"
                else:
                    price_info += f"📉 価格変動　：{price_diff:.2f}%\n"
                price_info += "━━━━━━━━━━━━━━━━━━━━━━━"
                
                print(price_info)
                self.logger.info(price_info)

                # 残高情報の表示
                usdt_balance, btc_holding = self.bybit.get_wallet_info()
                self.print_trade_info("情報", current_time, current_price, usdt_balance, btc_holding)

                # 取引ロジック
                if current_price < self.best_price * 0.999:  # 0.1%下落で買い
                    buy_msg = "買いシグナル検出。買い注文を実行中..."
                    print(buy_msg)
                    self.logger.info(buy_msg)
                    buy_amount_usdt = 50
                    btc_qty = buy_amount_usdt / current_price
                    if btc_qty < 0.001:
                        btc_qty = 0.001
                        adjust_msg = "最小注文数量の0.001 BTCに調整しました"
                        print(adjust_msg)
                        self.logger.info(adjust_msg)
                    
                    order = self.bybit.place_buy_order(btc_qty)
                    if order:
                        order_msg = f"買い注文実行: {order}"
                        print(order_msg)
                        self.logger.info(order_msg)
                        self.best_price = current_price

                elif current_price > self.best_price * 1.001:  # 0.1%上昇で売り
                    sell_msg = "売りシグナル検出。売り注文を実行中..."
                    print(sell_msg)
                    self.logger.info(sell_msg)
                    sell_amount_usdt = 50
                    btc_qty = sell_amount_usdt / current_price
                    if btc_qty < 0.001:
                        btc_qty = 0.001
                        adjust_msg = "最小注文数量の0.001 BTCに調整しました"
                        print(adjust_msg)
                        self.logger.info(adjust_msg)
                    
                    order = self.bybit.place_sell_order(btc_qty)
                    if order:
                        order_msg = f"売り注文実行: {order}"
                        print(order_msg)
                        self.logger.info(order_msg)
                        self.best_price = current_price

                time.sleep(3)

        except KeyboardInterrupt:
            exit_msg = "\nプログラムがCtrl+Cにより終了されました。安全に終了します。"
            print(exit_msg)
            self.logger.info(exit_msg)

def main():
    bot = TradingBot()
    bot.initialize()
    bot.execute_trade()

if __name__ == "__main__":
    main() 