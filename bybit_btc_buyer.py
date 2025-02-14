#Bybitで取引するためのプログラム
#テストネットを使用して、BTCの購入、売却、ポジション確認を行う


import os
import time
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

def get_server_time():
    """Bybitサーバーの時間を取得"""
    session = HTTP(testnet=True)
    time_resp = session.get_server_time()
    return int(time.time() * 1000)  # ローカル時間を使用

# テストネットのAPIクライアントを初期化
session = HTTP(
    testnet=True,
    api_key=os.getenv('BYBIT_TEST_API_KEY'),
    api_secret=os.getenv('BYBIT_TEST_SECRET'),
    recv_window=20000  # recv_windowを20秒に設定
)

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

def get_wallet_balance():
    """ウォレットの残高を取得"""
    try:
        wallet = session.get_wallet_balance(
            accountType="UNIFIED",
            coin="USDT"
        )
        if 'result' in wallet and 'list' in wallet['result']:
            for account in wallet['result']['list']:
                for coin in account['coin']:
                    if coin['coin'] == 'USDT':
                        return float(coin['walletBalance'])
        print("USDTの残高が見つかりませんでした")
        return None
    except Exception as e:
        print(f"残高取得中にエラーが発生しました: {e}")
        return None

def get_btc_wallet_balance():
    """ウォレットのBTC残高を取得"""
    try:
        wallet = session.get_wallet_balance(
            accountType="UNIFIED",
            coin="BTC"
        )
        if 'result' in wallet and 'list' in wallet['result']:
            for account in wallet['result']['list']:
                for coin in account['coin']:
                    if coin['coin'] == 'BTC':
                        return float(coin['walletBalance'])
        print("BTCの残高が見つかりませんでした")
        return None
    except Exception as e:
        print(f"BTC残高取得中にエラーが発生しました: {e}")
        return None

def place_order(quantity, side):
    """
    注文を実行する
    :param quantity: 市場注文の場合、BuyならUSDT金額(notional)、SellならBTC数量(qty)
    :param side: 'Buy' または 'Sell'
    """
    try:
        if side == "Buy":
            # For market buy orders on spot, use quoteQty (USDTの注文額)
            order = session.place_order(
                category="spot",
                symbol="BTCUSDT",
                side=side,
                orderType="Market",
                quoteQty=str(quantity)
            )
        else:
            # For sell orders, use qty (BTC数量)
            order = session.place_order(
                category="spot",
                symbol="BTCUSDT",
                side=side,
                orderType="Market",
                qty=str(quantity)
            )
        print(f"注文が成功しました: {order}")
        return order
    except Exception as e:
        print(f"注文中にエラーが発生しました: {e}")
        return None

def show_menu():
    """メニューを表示"""
    print("\n=== Bybit BTCトレーダー ===")
    print("1: BTCを購入")
    print("2: BTCを売却")
    print("3: ウォレットの確認")
    print("4: 終了")
    return input("選択してください (1-4): ")

def main():
    while True:
        try:
            choice = show_menu()
            
            if choice == "4":
                print("プログラムを終了します")
                break
                
            # 現在の価格とポジション情報を取得
            btc_price = get_btc_price()
            if btc_price is None:
                print("価格の取得に失敗しました")
                continue
                
            btc_balance = get_btc_wallet_balance()
            usdt_balance = get_wallet_balance()
            print(f"\n現在のBTC価格: {btc_price:,.2f} USDT")
            print(f"ウォレットのBTC残高: {btc_balance if btc_balance is not None else 0} BTC")
            print(f"ウォレットのUSDT残高: {usdt_balance if usdt_balance is not None else 0} USDT")
            
            if choice == "1":  # BTCを購入
                if usdt_balance is None:
                    continue
                
                print(f"USDTの残高: {usdt_balance:,.2f}")
                investment_amount = usdt_balance * 0.5
                # 最小注文額のチェック（例: 10 USDT以上）
                if investment_amount < 10:
                    investment_amount = 10
                    print("最小注文額 10 USDTに調整しました")
                estimated_btc = round(investment_amount / btc_price, 3)
                print(f"約 {estimated_btc} BTC(約 {investment_amount:,.2f} USDT分)を購入します...")
                order = place_order(investment_amount, "Buy")
            
            elif choice == "2":  # BTCを売却
                if btc_balance is None or btc_balance < 0.001:
                    print("売却可能なBTCがありません")
                    continue
                sell_quantity = btc_balance
                print(f"{sell_quantity} BTCを売却します...")
                order = place_order(sell_quantity, "Sell")
            
            elif choice == "3":  # ウォレットの確認
                print("ウォレットの残高:")
                print(f"BTC: {btc_balance if btc_balance is not None else 0} BTC")
                print(f"USDT: {usdt_balance if usdt_balance is not None else 0} USDT")
            
            else:
                print("無効な選択です。1-4の数字を入力してください。")
                
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            
        input("\nEnterキーを押して続行...")

if __name__ == "__main__":
    main() 