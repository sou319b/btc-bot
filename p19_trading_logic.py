"""
担当するクラス
Bybitとの通信
取引の実行
ログ記録
残高管理
取引間隔の制御
"""
import time
from datetime import datetime
import logging
from p19_bybit_handler import BybitHandler
from p19_strategy import TradingStrategy
import os

class TradingBot:
    def __init__(self):
        self.bybit = BybitHandler()
        self.strategy = TradingStrategy()
        self.initial_total = None
        self.best_price = None
        self.last_trade_time = None
        self.min_hold_time = 60  # ホールド時間を1分に延長
        self.info_interval = 5   # 情報表示の間隔を5秒に設定
        self.buy_count = 0   # 買い取引回数
        self.sell_count = 0  # 売り取引回数
        self.price_history = []  # 価格履歴を保存
        self.entry_price = None  # 購入価格
        self.btc_balance = 0     # BTC保有量
        self.usdt_balance = 0    # USDT残高
        self.min_start_balance = 10  # 開始に必要な最小USDT残高
        self.setup_logging()

    def setup_logging(self):
        """ロギングの設定を行う"""
        try:
            # logsディレクトリが存在しない場合は作成
            os.makedirs("logs", exist_ok=True)
            
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
            self.logger.info("ログ設定を初期化しました")
        except Exception as e:
            print(f"ログ設定の初期化に失敗しました: {e}")
            raise

    def print_trade_info(self, action, current_time, current_price, usdt_balance, btc_holding, trend_score, price_change_pct=None):
        total_assets = usdt_balance + btc_holding * current_price
        
        log_message = f"\n━━━━━━━━━━ 取引情報 ━━━━━━━━━━\n"
        log_message += f"📅 時刻　　　：{current_time}\n"
        log_message += f"💰 BTC価格　：{current_price:,.2f} USDT\n"
        if price_change_pct is not None:
            if price_change_pct > 0:
                log_message += f"📈 価格変動　：+{price_change_pct:.2f}%\n"
            else:
                log_message += f"📉 価格変動　：{price_change_pct:.2f}%\n"
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
        """初期化処理"""
        self.logger.info("初期化を開始します...")
        print("初期化を開始します...")
        
        # 初期価格の取得
        retry_count = 0
        max_retries = 3
        initial_price = None
        
        while initial_price is None and retry_count < max_retries:
            initial_price = self.bybit.get_btc_price()
            if initial_price is None:
                retry_count += 1
                self.logger.warning(f"価格取得に失敗しました。リトライ {retry_count}/{max_retries}")
                time.sleep(self.info_interval)
        
        if initial_price is None:
            error_msg = "初期価格の取得に失敗しました"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
        
        # 残高の初期化と確認
        usdt_balance, btc_balance = self.bybit.get_wallet_info()
        
        if btc_balance > 0:
            self.logger.warning(f"BTCの残高があります：{btc_balance} BTC")
            print(f"⚠️ 警告：BTCの残高があります：{btc_balance} BTC")
        
        if usdt_balance < self.min_start_balance:
            error_msg = f"USDT残高が不足しています：{usdt_balance} USDT"
            self.logger.error(error_msg)
            print(f"❌ エラー：{error_msg}")
            raise ValueError(f"取引開始には{self.min_start_balance} USDT以上の残高が必要です")
        
        self.usdt_balance = usdt_balance
        self.btc_balance = btc_balance
        self.initial_total = usdt_balance + btc_balance * initial_price
        self.best_price = initial_price
        
        # 価格履歴を初期化（同じ価格で初期化せず、少しずつ変動を付ける）
        variation = 0.0001  # 0.01%の変動
        self.price_history = [
            initial_price * (1 + (i - self.strategy.history_size/2) * variation)
            for i in range(self.strategy.history_size)
        ]
        
        self.logger.info("初期化が完了しました")
        print(f"\n━━━━━━━━━━ 初期状態 ━━━━━━━━━━")
        print(f"💫 初期BTC価格：{initial_price:,.2f} USDT")
        print(f"💰 初期総資産　：{self.initial_total:,.2f} USDT")
        print(f"💵 USDT残高　 ：{self.usdt_balance:,.2f} USDT")
        print(f"₿ BTC保有量　：{self.btc_balance:.6f} BTC")
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

                # 価格履歴を更新
                self.price_history.append(current_price)
                if len(self.price_history) > self.strategy.history_size:
                    self.price_history.pop(0)

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_timestamp = time.time()
                
                # 残高情報の更新
                self.usdt_balance, self.btc_balance = self.bybit.get_wallet_info()
                
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

                # トレンドとボラティリティを計算
                trend_score, volatility, price_change_pct = self.strategy.calculate_trend(self.price_history)
                
                self.print_trade_info("情報", current_time, current_price, self.usdt_balance, self.btc_balance, trend_score, price_change_pct)

                # 損切り・利確チェック
                if self.btc_balance > 0 and self.entry_price and self.strategy.should_close_position(current_price, self.entry_price):
                    sell_msg = "損切り/利確条件到達。売り注文を実行中..."
                    print(sell_msg)
                    self.logger.info(sell_msg)
                    
                    # BTCの数量を計算
                    sell_amount_btc = self.strategy.calculate_position_size(current_price, self.btc_balance * current_price)
                    
                    order = self.bybit.place_sell_order(sell_amount_btc)
                    if order:
                        self.last_trade_time = current_timestamp
                        self.sell_count += 1
                        self.entry_price = None
                        order_msg = f"売り注文実行: {order}"
                        print(order_msg)
                        self.logger.info(order_msg)
                    continue

                # 取引ロジック
                if self.strategy.should_buy(trend_score, volatility):  # 買いシグナル
                    if self.usdt_balance >= self.strategy.min_trade_amount:
                        # 最適な取引量を計算
                        trade_amount = self.strategy.calculate_optimal_trade_amount(
                            current_price, trend_score, volatility, self.usdt_balance
                        )
                        
                        if trade_amount >= self.strategy.min_trade_amount:
                            buy_msg = "下降トレンド検出。買い注文を実行中..."
                            print(buy_msg)
                            self.logger.info(buy_msg)
                            
                            # BTCの数量を計算
                            btc_qty = self.strategy.calculate_position_size(current_price, trade_amount)
                            
                            # 最終チェック
                            if btc_qty * current_price >= self.strategy.min_trade_amount:
                                order = self.bybit.place_buy_order(btc_qty)
                                if order:
                                    self.last_trade_time = current_timestamp
                                    self.buy_count += 1
                                    self.entry_price = current_price
                                    order_msg = f"買い注文実行: {order}"
                                    print(order_msg)
                                    self.logger.info(order_msg)
                            else:
                                skip_msg = "取引量が最小取引額を下回るため、取引をスキップします"
                                print(skip_msg)
                                self.logger.info(skip_msg)

                elif self.strategy.should_sell(trend_score) and self.btc_balance > 0:  # 売りシグナル
                    btc_value = self.btc_balance * current_price
                    if btc_value >= self.strategy.min_trade_amount:
                        sell_msg = "上昇トレンド検出。売り注文を実行中..."
                        print(sell_msg)
                        self.logger.info(sell_msg)
                        
                        # BTCの数量を計算
                        sell_amount_btc = self.strategy.calculate_position_size(current_price, btc_value)
                        
                        # 最終チェック
                        if sell_amount_btc * current_price >= self.strategy.min_trade_amount:
                            order = self.bybit.place_sell_order(sell_amount_btc)
                            if order:
                                self.last_trade_time = current_timestamp
                                self.sell_count += 1
                                self.entry_price = None
                                order_msg = f"売り注文実行: {order}"
                                print(order_msg)
                                self.logger.info(order_msg)
                        else:
                            skip_msg = "取引量が最小取引額を下回るため、取引をスキップします"
                            print(skip_msg)
                            self.logger.info(skip_msg)

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