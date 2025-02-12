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
            category="linear",
            symbol="BTCUSDT"
        )
        return float(ticker['result']['list'][0]['lastPrice'])
    except Exception as e:
        print(f"価格取得中にエラーが発生しました: {e}")
        return None

def get_position():
    """現在のポジション情報を取得"""
    try:
        position = session.get_positions(
            category="linear",
            symbol="BTCUSDT"
        )
        if position['result']['list']:
            size = float(position['result']['list'][0]['size'])
            side = position['result']['list'][0]['side']
            return size, side
        return 0, None
    except Exception as e:
        print(f"ポジション情報の取得中にエラーが発生しました: {e}")
        return 0, None

def place_order(quantity, side):
    """
    注文を実行する
    :param quantity: 取引量
    :param side: 'Buy' または 'Sell'
    """
    try:
        order = session.place_order(
            category="linear",
            symbol="BTCUSDT",
            side=side,
            orderType="Market",
            qty=str(quantity),
            reduceOnly=False,
            closeOnTrigger=False
        )
        print(f"注文が成功しました: {order}")
        return order
    except Exception as e:
        print(f"注文中にエラーが発生しました: {e}")
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

def show_menu():
    """メニューを表示"""
    print("\n=== Bybit BTCトレーダー ===")
    print("1: BTCを購入")
    print("2: BTCを売却")
    print("3: 現在のポジション確認")
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
                
            position_size, position_side = get_position()
            print(f"\n現在のBTC価格: {btc_price:,.2f} USDT")
            print(f"現在のポジション: {position_size} BTC ({position_side if position_side else '無し'})")
            
            if choice == "1":  # 購入
                usdt_balance = get_wallet_balance()
                if usdt_balance is None:
                    continue
                
                print(f"USDTの残高: {usdt_balance:,.2f}")
                investment_amount = usdt_balance * 0.5
                btc_quantity = round(investment_amount / btc_price, 3)
                
                if btc_quantity < 0.001:
                    btc_quantity = 0.001
                    print("最小注文数量の0.001 BTCに調整しました")
                
                if btc_quantity > 0:
                    print(f"{btc_quantity} BTCを購入します...")
                    order = place_order(btc_quantity, "Buy")
                else:
                    print("残高が不足しています")
                    
            elif choice == "2":  # 売却
                if position_size <= 0:
                    print("売却可能なBTCポジションがありません")
                    continue
                    
                print(f"現在の保有量: {position_size} BTC")
                sell_quantity = position_size
                if sell_quantity >= 0.001:
                    print(f"{sell_quantity} BTCを売却します...")
                    order = place_order(sell_quantity, "Sell")
                else:
                    print("売却可能な最小数量に満たないため、売却できません")
                    
            elif choice == "3":  # ポジション確認
                if position_size > 0:
                    value = position_size * btc_price
                    print(f"保有BTC: {position_size} BTC")
                    print(f"評価額: {value:,.2f} USDT")
                else:
                    print("現在ポジションはありません")
                    
            else:
                print("無効な選択です。1-4の数字を入力してください。")
                
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            
        input("\nEnterキーを押して続行...")

if __name__ == "__main__":
    main() 