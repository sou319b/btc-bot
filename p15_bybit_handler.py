import os
import logging
import time
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

class BybitHandler:
    def __init__(self):
        # .envファイルから環境変数を読み込む
        load_dotenv()
        
        # ロガーの設定
        self.logger = logging.getLogger(__name__)
        
        # サーバー時刻との差分を初期化
        self.time_offset = 0
        
        # 初期化時にサーバー時刻との同期を実行
        if not self._sync_time():
            self.logger.error("サーバー時刻との同期に失敗しました")
            return
            
        # Bybitテストネット用のAPIクライアントを初期化
        self.session = HTTP(
            testnet=True,
            api_key=os.getenv('BYBIT_TEST_API_KEY'),
            api_secret=os.getenv('BYBIT_TEST_SECRET'),
            recv_window=5000  # 5秒に設定（ドキュメント推奨値）
        )
        self.logger.info("Bybit APIクライアントを初期化しました")

    def _sync_time(self):
        """サーバー時刻との同期を行う"""
        try:
            # 時刻同期用の一時的なセッション
            temp_session = HTTP(testnet=True)
            server_time = temp_session.get_server_time()
            
            if 'result' not in server_time or 'timeSecond' not in server_time['result']:
                self.logger.error("サーバー時刻の取得に失敗しました")
                return False
                
            server_timestamp = int(server_time['result']['timeSecond'])
            local_timestamp = int(time.time())
            self.time_offset = server_timestamp - local_timestamp
            
            self.logger.info(f"サーバー時刻との差: {self.time_offset}秒")
            return True
            
        except Exception as e:
            self.logger.error(f"サーバー時刻同期エラー: {e}")
            return False

    def _get_timestamp(self):
        """補正済みのタイムスタンプを取得"""
        return int(time.time() + self.time_offset)

    def get_btc_price(self):
        """BTCの現在価格を取得する関数"""
        try:
            ticker = self.session.get_tickers(
                category="spot",
                symbol="BTCUSDT"
            )
            price = float(ticker['result']['list'][0]['lastPrice'])
            self.logger.debug(f"BTCの現在価格を取得: {price} USDT")
            return price
        except Exception as e:
            self.logger.error(f"価格取得中にエラーが発生: {e}")
            return None

    def get_wallet_info(self):
        """USDT残高とBTC保有量を取得する関数"""
        try:
            wallet = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            usdt_balance = 0.0
            if 'result' in wallet and 'list' in wallet['result']:
                for account in wallet['result']['list']:
                    if 'coin' in account:
                        for coin_info in account['coin']:
                            if coin_info['coin'] == 'USDT':
                                usdt_balance = float(coin_info['walletBalance'])
                                break
            self.logger.debug(f"USDT残高を取得: {usdt_balance} USDT")
        except Exception as e:
            self.logger.error(f"USDT残高取得エラー: {e}")
            usdt_balance = 0.0

        try:
            wallet = self.session.get_wallet_balance(accountType="UNIFIED", coin="BTC")
            btc_holding = 0.0
            if 'result' in wallet and 'list' in wallet['result']:
                for account in wallet['result']['list']:
                    if 'coin' in account:
                        for coin_info in account['coin']:
                            if coin_info['coin'] == 'BTC':
                                btc_holding = float(coin_info['walletBalance'])
                                break
            self.logger.debug(f"BTC保有量を取得: {btc_holding} BTC")
        except Exception as e:
            self.logger.error(f"BTC残高取得エラー: {e}")
            btc_holding = 0.0

        return usdt_balance, btc_holding

    def place_buy_order(self, qty):
        """買い注文を実行する関数"""
        try:
            order = self.session.place_order(
                category="spot",
                symbol="BTCUSDT",
                side="Buy",
                orderType="Market",
                qty=str(round(qty, 3))
            )
            self.logger.info(f"買い注文を実行: 数量={qty} BTC")
            return order
        except Exception as e:
            self.logger.error(f"買い注文エラー: {e}")
            return None

    def place_sell_order(self, qty):
        """売り注文を実行する関数"""
        try:
            order = self.session.place_order(
                category="spot",
                symbol="BTCUSDT",
                side="Sell",
                orderType="Market",
                qty=str(round(qty, 3))
            )
            self.logger.info(f"売り注文を実行: 数量={qty} BTC")
            return order
        except Exception as e:
            self.logger.error(f"売り注文エラー: {e}")
            return None 