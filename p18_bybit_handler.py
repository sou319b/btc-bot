# Bybit Unified Trading API Documentation:
# https://bybit-exchange.github.io/docs/unified/v3/spot/

import os
import logging
import time
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import math

class BybitHandler:
    def __init__(self):
        # .envファイルから環境変数を読み込む
        load_dotenv()
        
        # ロガーの設定
        self.logger = logging.getLogger(__name__)
        
        # APIキーとシークレットの検証
        api_key = os.getenv('BYBIT_TEST_API_KEY')
        api_secret = os.getenv('BYBIT_TEST_SECRET')
        
        if not api_key or not api_secret:
            error_msg = "APIキーまたはシークレットが設定されていません。.envファイルを確認してください。"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # サーバー時刻との差分を初期化
        self.time_offset = 0
        
        # 取引制限の設定
        self.MIN_BTC_QTY = 0.000100  # 最小BTC取引量
        self.MIN_USDT_VALUE = 5  # 最小注文金額（USDT）を5に設定
        
        # 初期化時にサーバー時刻との同期を実行
        if not self._sync_time():
            error_msg = "サーバー時刻との同期に失敗しました"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
            
        try:
            # Bybitテストネット用のAPIクライアントを初期化
            self.session = HTTP(
                testnet=True,
                api_key=api_key,
                api_secret=api_secret,
                recv_window=5000  # 5秒に設定（ドキュメント推奨値）
            )
            self.logger.info("Bybit APIクライアントを初期化しました")
        except Exception as e:
            error_msg = f"APIクライアントの初期化に失敗しました: {e}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)

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
            # UNIFIED口座の残高を取得
            spot_balance = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            usdt_balance = 0.0
            if 'result' in spot_balance and 'list' in spot_balance['result']:
                for account in spot_balance['result']['list']:
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
            # UNIFIED口座のBTC残高を取得
            btc_balance = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="BTC"
            )
            btc_holding = 0.0
            if 'result' in btc_balance and 'list' in btc_balance['result']:
                for account in btc_balance['result']['list']:
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

    def _round_btc(self, amount):
        """BTCの数量を5桁に丸める"""
        return "{:.5f}".format(float(amount))

    def place_buy_order(self, qty):
        """買い注文を実行する関数"""
        try:
            # 現在価格を取得して最小取引額をチェック
            current_price = self.get_btc_price()
            if current_price is None:
                self.logger.error("価格の取得に失敗しました")
                return None
                
            order_value = qty * current_price
            
            # 最小取引額のチェック
            if order_value < self.MIN_USDT_VALUE:
                self.logger.error(f"取引額が最小制限（{self.MIN_USDT_VALUE} USDT）を下回っています")
                return None

            # 取引直前の残高チェック
            usdt_balance, _ = self.get_wallet_info()
            if usdt_balance < order_value:
                self.logger.error(f"USDT残高不足: 必要額={order_value:.2f} USDT, 残高={usdt_balance:.2f} USDT")
                return None
            
            # USDTベースで注文を実行（金額を小数点以下2桁に丸める）
            rounded_value = "{:.2f}".format(order_value)
            
            order = self.session.place_order(
                category="spot",
                symbol="BTCUSDT",
                side="Buy",
                orderType="Market",
                qty=rounded_value,  # 丸めた金額を使用
                isQuoteQty=True,  # USDT金額での注文を指定
                orderFilter="ORDER"
            )
            self.logger.info(f"スポット買い注文を実行: 金額={rounded_value} USDT（想定数量: {qty} BTC）")
            return order
        except Exception as e:
            self.logger.error(f"買い注文エラー: {e}")
            return None

    def place_sell_order(self, qty):
        """売り注文を実行する関数"""
        try:
            # 現在価格を取得して最小取引額をチェック
            current_price = self.get_btc_price()
            if current_price is None:
                self.logger.error("価格の取得に失敗しました")
                return None
                
            order_value = qty * current_price
            
            # 最小取引額のチェック
            if order_value < self.MIN_USDT_VALUE:
                self.logger.error(f"取引額が最小制限（{self.MIN_USDT_VALUE} USDT）を下回っています")
                return None

            # 取引直前のBTC残高チェック
            _, btc_balance = self.get_wallet_info()
            if btc_balance < qty:
                self.logger.error(f"BTC残高不足: 必要数量={qty:.5f} BTC, 残高={btc_balance:.5f} BTC")
                return None
            
            # 数量を5桁に丸める
            rounded_qty = self._round_btc(qty)
            
            # スポット注文を実行
            order = self.session.place_order(
                category="spot",
                symbol="BTCUSDT",
                side="Sell",
                orderType="Market",
                qty=rounded_qty,
                orderFilter="ORDER"
            )
            self.logger.info(f"スポット売り注文を実行: 数量={rounded_qty} BTC（想定取引額: {order_value:.2f} USDT）")
            return order
        except Exception as e:
            self.logger.error(f"売り注文エラー: {e}")
            return None 