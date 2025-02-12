import os
import logging
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

class BybitHandler:
    def __init__(self):
        # .envファイルから環境変数を読み込む
        load_dotenv()
        
        # ロガーの設定
        self.logger = logging.getLogger(__name__)
        
        # Bybitテストネット用のAPIクライアントを初期化
        self.session = HTTP(
            testnet=True,
            api_key=os.getenv('BYBIT_TEST_API_KEY'),
            api_secret=os.getenv('BYBIT_TEST_SECRET'),
            recv_window=20000
        )
        self.logger.info("Bybit APIクライアントを初期化しました")

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
                category="linear",
                symbol="BTCUSDT",
                side="Buy",
                order_type="Market",
                qty=str(round(qty, 6)),
                time_in_force="GoodTillCancel"
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
                category="linear",
                symbol="BTCUSDT",
                side="Sell",
                order_type="Market",
                qty=str(round(qty, 6)),
                time_in_force="GoodTillCancel"
            )
            self.logger.info(f"売り注文を実行: 数量={qty} BTC")
            return order
        except Exception as e:
            self.logger.error(f"売り注文エラー: {e}")
            return None 