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
            # BTCの数量を計算（USDT額をBTC価格で割る）
            btc_price = get_btc_price()
            if btc_price is None:
                raise Exception("BTCの価格取得に失敗しました")
            btc_quantity = round(float(quantity) / btc_price, 3)  # 小数点以下3桁に制限
            
            # 最小取引量のチェック
            if btc_quantity < 0.001:  # 最小取引量を0.001 BTCに設定
                raise Exception("取引量が小さすぎます。より大きな金額で取引してください。")
            
            # 指値注文の価格を現在価格より少し高めに設定（約0.1%）
            limit_price = round(btc_price * 1.001, 2)
            
            order = session.place_order(
                category="spot",
                symbol="BTCUSDT",
                side=side,
                orderType="Limit",
                qty=str(btc_quantity),  # BTCの数量を指定
                price=str(limit_price),  # 指値価格を指定
                timeInForce="GTC"  # Good Till Cancel
            )
        else:
            # For sell orders, use qty (BTC数量)
            # 小数点以下3桁に制限
            quantity = round(float(quantity), 3)
            if quantity < 0.001:  # 最小取引量を0.001 BTCに設定
                raise Exception("取引量が小さすぎます。最小取引量は0.001 BTCです。")
            
            # 指値注文の価格を現在価格より少し低めに設定（約0.1%）
            btc_price = get_btc_price()
            limit_price = round(btc_price * 0.999, 2)
            
            order = session.place_order(
                category="spot",
                symbol="BTCUSDT",
                side=side,
                orderType="Limit",
                qty=str(quantity),
                price=str(limit_price),  # 指値価格を指定
                timeInForce="GTC"  # Good Till Cancel
            )
        print(f"注文が成功しました: {order}")
        return order
    except Exception as e:
        print(f"注文中にエラーが発生しました: {e}")
        return None

def get_trading_limits(balance, btc_price, side):
    """
    取引可能な範囲を計算する
    :param balance: 残高（USDTまたはBTC）
    :param btc_price: BTCの現在価格
    :param side: 'Buy' または 'Sell'
    :return: (最小取引額, 最大取引額)
    """
    if side == "Buy":
        min_btc = 0.001  # 最小BTC取引量を0.001に設定
        min_trade = max(10, min_btc * btc_price)  # 最小取引額を計算（10 USDTまたはBTC最小量相当のUSDT）
        max_trade = min(balance, 100000)  # 残高か100000USDTの小さい方
        return min_trade, max_trade
    else:
        min_trade = 0.001  # 最小BTC取引量を0.001に設定
        max_trade = min(balance, 100)  # 残高か100BTCの小さい方
        return min_trade, max_trade

def get_trade_amount(min_amount, max_amount, side):
    """
    ユーザーに取引額を入力してもらう
    """
    while True:
        if side == "Buy":
            print(f"\n取引可能範囲: {min_amount:.2f} USDT から {max_amount:.2f} USDT")
            amount_str = input("取引するUSDT額を入力してください: ")
        else:
            print(f"\n取引可能範囲: {min_amount:.4f} BTC から {max_amount:.4f} BTC")
            amount_str = input("取引するBTC量を入力してください: ")
        
        try:
            amount = float(amount_str)
            if min_amount <= amount <= max_amount:
                return amount
            else:
                print(f"取引可能範囲内で入力してください")
        except ValueError:
            print("有効な数値を入力してください")

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
                min_btc = 0.001  # 最小BTC取引量を0.001に設定
                min_usdt = min_btc * btc_price
                if usdt_balance is None or usdt_balance < min_usdt:
                    print(f"取引に必要な残高（最小{min_usdt:.2f} USDT）がありません")
                    continue
                
                min_trade, max_trade = get_trading_limits(usdt_balance, btc_price, "Buy")
                investment_amount = get_trade_amount(min_trade, max_trade, "Buy")
                estimated_btc = round(investment_amount / btc_price, 3)
                print(f"約 {estimated_btc} BTC(約 {investment_amount:,.2f} USDT分)を購入します...")
                order = place_order(investment_amount, "Buy")
            
            elif choice == "2":  # BTCを売却
                if btc_balance is None or btc_balance < 0.001:  # 最小取引量を0.001 BTCに設定
                    print("売却可能なBTCがありません（最小0.001 BTC）")
                    continue
                
                min_trade, max_trade = get_trading_limits(btc_balance, btc_price, "Sell")
                sell_quantity = get_trade_amount(min_trade, max_trade, "Sell")
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