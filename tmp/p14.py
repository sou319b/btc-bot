import os
import time
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# Bybitテストネット用のAPIクライアントを初期化
session = HTTP(
    testnet=True,
    api_key=os.getenv('BYBIT_TEST_API_KEY'),
    api_secret=os.getenv('BYBIT_TEST_SECRET'),
    recv_window=20000
)


def get_btc_price():
    """BTCの現在価格を取得する関数"""
    try:
        ticker = session.get_tickers(
            category="spot",
            symbol="BTCUSDT"
        )
        return float(ticker['result']['list'][0]['lastPrice'])
    except Exception as e:
        print(f"価格取得中にエラーが発生: {e}")
        return None


def print_trade_info(action, current_time, current_price, usdt_balance, btc_holding, initial_total=None):
    total_assets = usdt_balance + btc_holding * current_price
    
    print("\n━━━━━━━━━━ 取引情報 ━━━━━━━━━━")
    print(f"📅 時刻　　　：{current_time}")
    print(f"💰 BTC価格　：{current_price:,.2f} USDT")
    print(f"💵 USDT残高 ：{usdt_balance:,.2f} USDT")
    print(f"₿ BTC保有量：{btc_holding:.6f} BTC")
    print(f"📊 総資産額 ：{total_assets:,.2f} USDT")
    
    if initial_total is not None:
        profit = total_assets - initial_total
        profit_percentage = (profit / initial_total) * 100
        if profit >= 0:
            print(f"💹 現在の利益：+{profit:,.2f} USDT (+{profit_percentage:.2f}%)")
        else:
            print(f"📉 現在の損失：{profit:,.2f} USDT ({profit_percentage:.2f}%)")
    
    if action != "情報":
        print(f"📈 取引種別 ：{action}")
    print("━━━━━━━━━━━━━━━━━━━━━━━")


def get_wallet_info():
    """USDT残高とBTC保有量を取得する関数"""
    try:
        wallet = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        usdt_balance = 0.0
        if 'result' in wallet and 'list' in wallet['result']:
            for account in wallet['result']['list']:
                if 'coin' in account:
                    for coin_info in account['coin']:
                        if coin_info['coin'] == 'USDT':
                            usdt_balance = float(coin_info['walletBalance'])
                            break
    except Exception as e:
        print(f"USDT残高取得エラー: {e}")
        usdt_balance = 0.0

    try:
        wallet = session.get_wallet_balance(accountType="UNIFIED", coin="BTC")
        btc_holding = 0.0
        if 'result' in wallet and 'list' in wallet['result']:
            for account in wallet['result']['list']:
                if 'coin' in account:
                    for coin_info in account['coin']:
                        if coin_info['coin'] == 'BTC':
                            btc_holding = float(coin_info['walletBalance'])
                            break
    except Exception as e:
        print(f"BTC残高取得エラー: {e}")
        btc_holding = 0.0

    return usdt_balance, btc_holding


def main():
    print("初期価格を取得中...")
    initial_price = None
    while initial_price is None:
        initial_price = get_btc_price()
        if initial_price is None:
            time.sleep(1)
    
    # 初期資産を計算
    initial_usdt, initial_btc = get_wallet_info()
    initial_total = initial_usdt + initial_btc * initial_price
    
    print(f"\n━━━━━━━━━━ 初期状態 ━━━━━━━━━━")
    print(f"💫 初期BTC価格：{initial_price:,.2f} USDT")
    print(f"💰 初期総資産　：{initial_total:,.2f} USDT")
    print("━━━━━━━━━━━━━━━━━━━━━━━\n")

    best_price = initial_price

    try:
        while True:
            base_price = get_btc_price()
            if base_price is None:
                print("価格の取得に失敗しました。再試行します...")
                time.sleep(1)
                continue

            current_price = base_price
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            price_diff = ((current_price - best_price) / best_price) * 100
            
            # 価格変動情報の表示
            print("\n━━━━━━━━━━ 価格情報 ━━━━━━━━━━")
            print(f"📅 時刻　　　：{current_time}")
            print(f"💰 現在価格　：{current_price:,.2f} USDT")
            if price_diff > 0:
                print(f"📈 価格変動　：+{price_diff:.2f}%")
            else:
                print(f"📉 価格変動　：{price_diff:.2f}%")
            print("━━━━━━━━━━━━━━━━━━━━━━━")

            # 常にUSDT残高、BTC保有量、総資産額を表示
            usdt_balance, btc_holding = get_wallet_info()
            print_trade_info("情報", current_time, current_price, usdt_balance, btc_holding, initial_total)

            # 買い条件：現在価格が基準価格の0.1%下落した場合
            if current_price < best_price * 0.999:
                print("買いシグナル検出。買い注文を実行中...")
                try:
                    buy_amount_usdt = 50  # 50 USDT分の買い注文
                    btc_qty = buy_amount_usdt / current_price
                    if btc_qty < 0.001:
                        btc_qty = 0.001
                        print("最小注文数量の0.001 BTCに調整しました")
                    qty_str = str(round(btc_qty, 6))
                    order = session.place_order(
                        category="linear",
                        symbol="BTCUSDT",
                        side="Buy",
                        order_type="Market",
                        qty=qty_str,
                        time_in_force="GoodTillCancel"
                    )
                    print("買い注文実行:", order)
                    best_price = current_price
                except Exception as e:
                    print(f"買い注文エラー: {e}")

            # 売り条件：現在価格が基準価格の0.1%上昇した場合
            elif current_price > best_price * 1.001:
                print("売りシグナル検出。売り注文を実行中...")
                try:
                    sell_amount_usdt = 50  # 50 USDT分の売り注文
                    btc_qty = sell_amount_usdt / current_price
                    if btc_qty < 0.001:
                        btc_qty = 0.001
                        print("最小注文数量の0.001 BTCに調整しました")
                    qty_str = str(round(btc_qty, 6))
                    order = session.place_order(
                        category="linear",
                        symbol="BTCUSDT",
                        side="Sell",
                        order_type="Market",
                        qty=qty_str,
                        time_in_force="GoodTillCancel"
                    )
                    print("売り注文実行:", order)
                    best_price = current_price
                except Exception as e:
                    print(f"売り注文エラー: {e}")

            time.sleep(3)  # 更新間隔を3秒に設定
    except KeyboardInterrupt:
        print("\nプログラムがCtrl+Cにより終了されました。安全に終了します。")


if __name__ == "__main__":
    main() 