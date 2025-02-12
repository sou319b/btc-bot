import os
import time
import random
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
        ticker = session.get_tickers(category="linear", symbol="BTCUSDT")
        return float(ticker['result']['list'][0]['lastPrice'])
    except Exception as e:
        print(f"価格取得中にエラーが発生: {e}")
        return None


def print_trade_info(action, current_time, current_price, jpy_balance, btc_holding):
    total_assets = jpy_balance + btc_holding * current_price
    print("----------------")
    print(f"現在の時刻：{current_time}")
    print(f"現在のBTC価格：{current_price:.2f}")
    print(f"JPY残高: {jpy_balance:.2f}")
    print(f"BTC保有量：{btc_holding:.6f}")
    print(f"総資産額：{total_assets:.2f}")
    print(f"買/売：{action}")
    print("----------------\n")


def main():
    # 初期資金100000円
    jpy_balance = 100000.0
    btc_holding = 0.0

    # 初期価格の取得
    print("初期価格を取得中...")
    initial_price = None
    while initial_price is None:
        initial_price = get_btc_price()
        if initial_price is None:
            time.sleep(1)
    print(f"初期BTC価格: {initial_price:.2f}\n")

    # best_priceを初期化（取引基準の価格）
    best_price = initial_price

    try:
        while True:
            base_price = get_btc_price()
            if base_price is None:
                print("価格の取得に失敗しました。再試行します...")
                time.sleep(1)
                continue

            # ランダムな変動を付与（±0.2%以内の変動）
            current_price = base_price * (1 + (random.random() * 0.004 - 0.002))

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # best_priceとの価格変動（％）を表示
            price_diff = ((current_price - best_price) / best_price) * 100
            print(f"価格変動: {price_diff:+.2f}%")

            # 買い条件：現在価格がbest_priceの0.1%下落し、JPY残高がある場合
            if current_price < best_price * 0.999 and jpy_balance > 0:
                buy_amount = min(jpy_balance, 50000)  # 最大5万円分購入
                btc_bought = buy_amount / current_price
                jpy_balance -= buy_amount
                btc_holding += btc_bought
                best_price = current_price
                print_trade_info("買", current_time, current_price, jpy_balance, btc_holding)

            # 売り条件：現在価格がbest_priceの0.1%上昇し、BTC保有がある場合
            elif current_price > best_price * 1.001 and btc_holding > 0:
                sell_amount = btc_holding * 0.5  # 保有BTCの50%売却
                jpy_balance += sell_amount * current_price
                btc_holding -= sell_amount
                best_price = current_price
                print_trade_info("売", current_time, current_price, jpy_balance, btc_holding)

            time.sleep(3)  # 更新間隔を3秒に設定
    except KeyboardInterrupt:
        print("\nプログラムがCtrl+Cにより終了されました。安全に終了します。")


if __name__ == "__main__":
    main() 