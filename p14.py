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
        now = int(time.time())
        # 直近の完了した1分足の開始時刻を取得
        candle_end = (now // 60) * 60  
        end_time = candle_end * 1000
        start_time = (candle_end - 60) * 1000

        kline = session.get_mark_price_kline(
            category="linear",
            symbol="BTCUSDT",
            interval=1,
            start=start_time,
            end=end_time,
            limit=1
        )

        if kline['result']['list']:
            # kline のフォーマット: [timestamp, open, high, low, close]
            return float(kline['result']['list'][0][4])
        else:
            print("klineデータが空です")
            return None
    except Exception as e:
        print(f"価格取得中にエラーが発生: {e}")
        return None


def print_trade_info(action, current_time, current_price, usdt_balance, btc_holding):
    total_assets = usdt_balance + btc_holding * current_price
    print("----------------")
    print(f"現在の時刻：{current_time}")
    print(f"現在のBTC価格：{current_price:.2f}")
    print(f"USDT残高: {usdt_balance:.2f}")
    print(f"BTC保有量：{btc_holding:.6f}")
    print(f"総資産額：{total_assets:.2f}")
    print(f"買/売：{action}")
    print("----------------\n")


def get_wallet_info():
    """USDT残高とBTC保有量を取得する関数"""
    try:
        wallet_info = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        result = wallet_info.get('result', {})
        usdt_balance = 0.0
        if 'list' in result:
            for asset in result['list']:
                if asset.get('coin') == 'USDT':
                    usdt_balance = float(asset.get('available_balance', 0))
                    break
        else:
            usdt_balance = float(result.get('USDT', {}).get('available_balance', 0))
    except Exception as e:
        print(f"ウォレット情報取得エラー: {e}")
        usdt_balance = 0.0

    try:
        pos_info = session.get_positions(category="linear", symbol="BTCUSDT")
        btc_holding = 0.0
        result = pos_info.get('result', {})
        if 'list' in result:
            for pos in result['list']:
                btc_holding += abs(float(pos.get('size', 0)))
        else:
            btc_holding = 0.0
    except Exception as e:
        print(f"ポジション情報取得エラー: {e}")
        btc_holding = 0.0

    return usdt_balance, btc_holding


def main():
    print("初期価格を取得中...")
    initial_price = None
    while initial_price is None:
        initial_price = get_btc_price()
        if initial_price is None:
            time.sleep(1)
    print(f"初期BTC価格: {initial_price:.2f}\n")

    best_price = initial_price

    try:
        while True:
            base_price = get_btc_price()
            if base_price is None:
                print("価格の取得に失敗しました。再試行します...")
                time.sleep(1)
                continue

            # 実際の市場価格を使用
            current_price = base_price
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            price_diff = ((current_price - best_price) / best_price) * 100
            print("----------------")
            print(f"現在の時刻：{current_time}")
            print(f"現在のBTC価格：{current_price:.2f}")
            print(f"基準価格からの変動: {price_diff:+.2f}%")
            print("----------------")

            # 常にUSDT残高、BTC保有量、総資産額を表示
            usdt_balance, btc_holding = get_wallet_info()
            print_trade_info("情報", current_time, current_price, usdt_balance, btc_holding)

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