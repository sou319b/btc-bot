import time
from datetime import datetime
import logging
from p14_bybit_handler import BybitHandler

class TradingBot:
    def __init__(self):
        self.bybit = BybitHandler()
        self.initial_total = None
        self.best_price = None
        self.last_trade_time = None
        self.min_hold_time = 60  # 最低1分間のホールド時間
        self.info_interval = 10   # 情報表示の間隔（秒）
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

    def print_trade_info(self, action, current_time, current_price, usdt_balance, btc_holding, price_diff=None):
        total_assets = usdt_balance + btc_holding * current_price
        
        log_message = f"\n━━━━━━━━━━ 取引情報 ━━━━━━━━━━\n"
        log_message += f"📅 時刻　　　：{current_time}\n"
        log_message += f"💰 BTC価格　：{current_price:,.2f} USDT\n"
        if price_diff is not None:
            if price_diff > 0:
                log_message += f"📈 価格変動　：+{price_diff:.2f}%\n"
            else:
                log_message += f"📉 価格変動　：{price_diff:.2f}%\n"
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
                time.sleep(self.info_interval)
        
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
                    time.sleep(self.info_interval)
                    continue

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_timestamp = time.time()
                
                # ホールド時間のチェック
                if self.last_trade_time:
                    time_since_last_trade = current_timestamp - self.last_trade_time
                    if time_since_last_trade < self.min_hold_time:
                        remaining_time = int(self.min_hold_time - time_since_last_trade)
                        hold_msg = f"前回の取引から{remaining_time}秒待機中..."
                        print(hold_msg)
                        self.logger.info(hold_msg)
                        time.sleep(self.info_interval)
                        continue

                price_diff = ((current_price - self.best_price) / self.best_price) * 100

                # 残高情報の表示
                usdt_balance, btc_holding = self.bybit.get_wallet_info()
                self.print_trade_info("情報", current_time, current_price, usdt_balance, btc_holding, price_diff)

                # 取引ロジック
                if current_price < self.best_price * 0.995:  # 0.5%下落で買い
                    if usdt_balance > 0:
                        buy_msg = "買いシグナル検出。買い注文を実行中..."
                        print(buy_msg)
                        self.logger.info(buy_msg)
                        
                        # 下落率に応じて取引量を調整
                        drop_percentage = abs(price_diff)
                        if drop_percentage > 1.0:  # 1%以上の下落
                            buy_amount_usdt = min(usdt_balance, 100000)  # より大きな取引
                        else:
                            buy_amount_usdt = min(usdt_balance, 50000)
                        
                        btc_qty = round(buy_amount_usdt / current_price, 3)
                        if btc_qty < 0.001:
                            btc_qty = 0.001
                            adjust_msg = "最小注文数量の0.001 BTCに調整しました"
                            print(adjust_msg)
                            self.logger.info(adjust_msg)
                        
                        order = self.bybit.place_buy_order(btc_qty)
                        if order:
                            self.last_trade_time = current_timestamp
                            order_msg = f"買い注文実行: {order}"
                            print(order_msg)
                            self.logger.info(order_msg)
                            self.best_price = current_price
                            time.sleep(self.info_interval)  # 注文後の待機時間を追加

                elif current_price > self.best_price * 1.005:  # 0.5%上昇で売り
                    if btc_holding > 0:
                        sell_msg = "売りシグナル検出。売り注文を実行中..."
                        print(sell_msg)
                        self.logger.info(sell_msg)
                        
                        # 上昇率に応じて取引量を調整
                        rise_percentage = price_diff
                        if rise_percentage > 1.0:  # 1%以上の上昇
                            btc_qty = round(btc_holding * 0.75, 3)  # より大きな取引
                        else:
                            btc_qty = round(btc_holding * 0.5, 3)
                        
                        if btc_qty < 0.001:
                            btc_qty = 0.001
                            adjust_msg = "最小注文数量の0.001 BTCに調整しました"
                            print(adjust_msg)
                            self.logger.info(adjust_msg)
                        
                        order = self.bybit.place_sell_order(btc_qty)
                        if order:
                            self.last_trade_time = current_timestamp
                            order_msg = f"売り注文実行: {order}"
                            print(order_msg)
                            self.logger.info(order_msg)
                            self.best_price = current_price
                            time.sleep(self.info_interval)  # 注文後の待機時間を追加

                time.sleep(self.info_interval)

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