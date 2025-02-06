import ccxt
import config
from typing import Dict, Any

class ExchangeHandler:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'apiKey': config.EXCHANGE_API_KEY,
            'secret': config.EXCHANGE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'testnet': config.TESTNET
            }
        })
        
    def get_balance(self) -> Dict[str, float]:
        """口座残高の取得"""
        try:
            balance = self.exchange.fetch_balance()
            return {
                'total': balance['total']['USDT'],
                'free': balance['free']['USDT'],
                'used': balance['used']['USDT']
            }
        except Exception as e:
            print(f"残高取得エラー: {e}")
            return {}

    def place_order(self, side: str, amount: float, price: float = None) -> Dict[str, Any]:
        """注文の発行"""
        try:
            order_type = 'market' if price is None else 'limit'
            order = self.exchange.create_order(
                symbol=config.TRADING_PAIR,
                type=order_type,
                side=side,
                amount=amount,
                price=price
            )
            return order
        except Exception as e:
            print(f"注文エラー: {e}")
            return {}

    def get_position(self) -> Dict[str, Any]:
        """現在のポジション情報の取得"""
        try:
            positions = self.exchange.fetch_positions([config.TRADING_PAIR])
            return positions[0] if positions else {}
        except Exception as e:
            print(f"ポジション取得エラー: {e}")
            return {}

    def fetch_ohlcv(self, timeframe: str = '1m', limit: int = 100) -> list:
        """価格データの取得"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=config.TRADING_PAIR,
                timeframe=timeframe,
                limit=limit
            )
            return ohlcv
        except Exception as e:
            print(f"価格データ取得エラー: {e}")
            return []
