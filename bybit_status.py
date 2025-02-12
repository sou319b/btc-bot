"""
Bybitのapiを使い残高、BTC価格を表示する
正しく動作する
   
"""

import os
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# テストネットのAPIクライアントを初期化
session = HTTP(
    testnet=True,
    api_key=os.getenv('BYBIT_TEST_API_KEY'),
    api_secret=os.getenv('BYBIT_TEST_SECRET'),
    recv_window=20000
)


def get_balance(coin):
    """指定されたcoin(BTCまたはUSDT)の残高を取得"""
    try:
        wallet = session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        if 'result' in wallet and 'list' in wallet['result']:
            for account in wallet['result']['list']:
                for coin_info in account['coin']:
                    if coin_info['coin'] == coin:
                        return float(coin_info['walletBalance'])
        print(f"{coin}の残高が見つかりませんでした")
        return None
    except Exception as e:
        print(f"{coin}残高取得中にエラーが発生しました: {e}")
        return None


def get_btc_price():
    """BTCの現在価格を取得"""
    try:
        ticker = session.get_tickers(
            category="spot",
            symbol="BTCUSDT"
        )
        return float(ticker['result']['list'][0]['lastPrice'])
    except Exception as e:
        print(f"価格取得中にエラーが発生しました: {e}")
        return None


def main():
    btc_balance = get_balance("BTC")
    usdt_balance = get_balance("USDT")
    btc_price = get_btc_price()
    
    print("\n=== Bybit Testnet 残高情報 ===")
    if btc_balance is not None:
        print(f"BTCの残高: {btc_balance}")
    else:
        print("BTCの残高の取得に失敗しました")
        
    if usdt_balance is not None:
        print(f"USDTの残高: {usdt_balance}")
    else:
        print("USDTの残高の取得に失敗しました")

    print("\n=== BTCの価格情報 ===")
    if btc_price is not None:
        print(f"BTCの価格: {btc_price}")
    else:
        print("BTCの価格の取得に失敗しました")


if __name__ == "__main__":
    main()
