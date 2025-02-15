import time
from datetime import datetime
import logging
from p16_bybit_handler import BybitHandler
import math

class TradingBot:
    def __init__(self):
        self.bybit = BybitHandler()
        self.initial_total = None
        self.best_price = None
        self.last_trade_time = None
        self.min_hold_time = 30  # ホールド時間を30秒に短縮
        self.info_interval = 5   # 情報表示の間隔を5秒に短縮
        self.min_trade_amount = 5  # 最小取引額（USDT）
        self.max_trade_amount = 50  # 最大取引額を50 USDTに設定
        self.buy_count = 0   # 買い取引回数
        self.sell_count = 0  # 売り取引回数
        self.price_history = []  # 価格履歴を保存
        self.history_size = 12   # 1分間の価格履歴（5秒×12）
        self.setup_logging()

    def setup_logging(self):
        """ロギングの設定を行う"""
        log_filename = f"logs/trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        log_message += f"🔄 取引回数 ：買い{self.buy_count}回 / 売り{self.sell_count}回\n"
        
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
        
        # 価格履歴を初期化
        self.price_history = [initial_price] * self.history_size
        
        print(f"\n━━━━━━━━━━ 初期状態 ━━━━━━━━━━")
        print(f"💫 初期BTC価格：{initial_price:,.2f} USDT")
        print(f"💰 初期総資産　：{self.initial_total:,.2f} USDT")
        print("━━━━━━━━━━━━━━━━━━━━━━━\n")

    def calculate_trend(self):
        """価格トレンドを計算"""
        if len(self.price_history) < 2:
            return 0
        
        # 直近の価格変動率を計算
        short_term_change = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2] * 100
        
        # 1分間の価格変動率を計算
        long_term_change = (self.price_history[-1] - self.price_history[0]) / self.price_history[0] * 100
        
        # トレンドスコアを計算（短期と長期の変動を組み合わせ）
        trend_score = short_term_change * 0.7 + long_term_change * 0.3
        return trend_score

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

                # 価格履歴を更新
                self.price_history.append(current_price)
                if len(self.price_history) > self.history_size:
                    self.price_history.pop(0)

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

                # トレンドスコアを計算
                trend_score = self.calculate_trend()
                
                # 残高情報の取得
                usdt_balance, btc_holding = self.bybit.get_wallet_info()
                self.print_trade_info("情報", current_time, current_price, usdt_balance, btc_holding, trend_score)

                # 取引ロジック
                if trend_score < -0.01:  # 下降トレンドで買い
                    if usdt_balance >= self.min_trade_amount:
                        buy_msg = "下降トレンド検出。買い注文を実行中..."
                        print(buy_msg)
                        self.logger.info(buy_msg)
                        
                        # トレンドの強さに応じて取引量を調整
                        trend_strength = abs(trend_score)
                        if trend_strength > 0.3:  # 強い下降トレンド
                            buy_amount_usdt = min(usdt_balance, self.max_trade_amount)
                        else:
                            buy_amount_usdt = min(usdt_balance, self.max_trade_amount * 0.7)
                        
                        # 最小取引額以上になるように調整
                        buy_amount_usdt = max(buy_amount_usdt, self.min_trade_amount)
                        
                        # BTCの数量を計算
                        btc_qty = buy_amount_usdt / current_price
                        btc_qty = math.ceil(btc_qty * 100000) / 100000  # 5桁に丸める
                        
                        order = self.bybit.place_buy_order(btc_qty)
                        if order:
                            self.last_trade_time = current_timestamp
                            self.buy_count += 1
                            self.best_price = current_price
                            order_msg = f"買い注文実行: {order}"
                            print(order_msg)
                            self.logger.info(order_msg)
                    else:
                        insufficient_msg = f"USDT残高が最小取引額（{self.min_trade_amount} USDT）未満のため、取引をスキップします"
                        print(insufficient_msg)
                        self.logger.info(insufficient_msg)

                elif trend_score > 0.01:  # 上昇トレンドで売り
                    if btc_holding > 0:
                        btc_value = btc_holding * current_price
                        if btc_value >= self.min_trade_amount:
                            sell_msg = "上昇トレンド検出。売り注文を実行中..."
                            print(sell_msg)
                            self.logger.info(sell_msg)
                            
                            # トレンドの強さに応じて取引量を調整
                            trend_strength = abs(trend_score)
                            if trend_strength > 0.3:  # 強い上昇トレンド
                                sell_amount_btc = btc_holding  # 全量売却
                            else:
                                sell_amount_btc = btc_holding * 0.7  # 70%売却
                            
                            # 最小取引額を確保
                            min_btc_amount = self.min_trade_amount / current_price
                            sell_amount_btc = max(sell_amount_btc, min_btc_amount)
                            
                            # 保有量を超えないように調整
                            sell_amount_btc = min(sell_amount_btc, btc_holding)
                            
                            order = self.bybit.place_sell_order(sell_amount_btc)
                            if order:
                                self.last_trade_time = current_timestamp
                                self.sell_count += 1
                                self.best_price = current_price
                                order_msg = f"売り注文実行: {order}"
                                print(order_msg)
                                self.logger.info(order_msg)
                        else:
                            insufficient_msg = f"BTC保有量の価値が最小取引額（{self.min_trade_amount} USDT）未満のため、取引をスキップします"
                            print(insufficient_msg)
                            self.logger.info(insufficient_msg)
                    else:
                        no_btc_msg = "BTC保有量が0のため、売り注文をスキップします"
                        print(no_btc_msg)
                        self.logger.info(no_btc_msg)

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